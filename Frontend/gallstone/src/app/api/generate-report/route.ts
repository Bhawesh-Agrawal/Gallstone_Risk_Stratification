import { NextRequest, NextResponse } from "next/server";
import { jsPDF } from "jspdf";
import { randomUUID } from "crypto";

// =================================================================================
// 1. IN-MEMORY JOB STORE & TYPES
// =================================================================================

// NEW: Added patientDetails to the request interface
interface ReportRequest {
  patientDetails: {
    name: string;
    address: string;
    phone: string;
  };
  prediction_label: "Positive" | "Negative";
  probability: { "positive (class 0)": string; "negative (class 1)": string };
  shap_analysis: {
    base_value_for_positive_class: number;
    base_value_for_negative_class: number;
    top_contributors_to_positive: Record<string, string>;
    top_contributors_to_negative: Record<string, string>;
    plots: { waterfall_plot: string; bar_summary_plot: string };
  };
}

interface Job {
  status: "pending" | "complete" | "failed";
  pdf?: string; // Base64 PDF string
  error?: string;
}

const jobStore = new Map<string, Job>();

// =================================================================================
// 2. CORE LOGIC (Gemini API Call and PDF Generation)
// =================================================================================

const generateMarkdownReport = async (data: ReportRequest): Promise<string> => {
  const { prediction_label, probability, shap_analysis } = data;
  
  // UPDATED: The prompt no longer needs the instruction to include the plot image.
  const prompt = `
    Please act as a medical data analyst. Generate a comprehensive but easy-to-understand report in Markdown format based on the following gallstone risk prediction data.

    **Prediction Summary:**
    - The model predicts: ${prediction_label}
    - Probability of being Positive (Class 0): ${(parseFloat(probability["positive (class 0)"]) * 100).toFixed(2)}%
    - Probability of being Negative (Class 1): ${(parseFloat(probability["negative (class 1)"]) * 100).toFixed(2)}%

    **Detailed SHAP Analysis:**
    - Base value for the positive class: ${shap_analysis.base_value_for_positive_class}
    - Top contributors pushing the prediction towards POSITIVE:
      ${Object.entries(shap_analysis.top_contributors_to_positive).map(([key, value]) => `- **${key}**: ${value}`).join("\n")}
    - Top contributors pushing the prediction towards NEGATIVE:
      ${Object.entries(shap_analysis.top_contributors_to_negative).map(([key, value]) => `- **${key}**: ${value}`).join("\n")}

    **Instructions for the report:**
    1.  Start with a clear summary of the prediction.
    2.  Explain what SHAP values mean in simple terms.
    3.  Discuss the "Top contributors to positive" and explain why they might increase the risk.
    4.  Discuss the "Top contributors to negative" and explain why they might be protective.
    5.  Conclude with a general disclaimer that this is not medical advice.
  `;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 180000); // 3-minute timeout

  try {
    const geminiRes = await fetch(
      "https://codexbhawesh-gallstone.hf.space/gemini-chat",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: prompt }),
        signal: controller.signal,
      }
    );

    if (!geminiRes.ok) {
      const errorBody = await geminiRes.text();
      throw new Error(`Gemini API request failed with status ${geminiRes.status}: ${errorBody}`);
    }

    const geminiData = await geminiRes.json();
    if (!geminiData.response) {
      throw new Error("Invalid response structure from Gemini API.");
    }
    return geminiData.response;
  } finally {
    clearTimeout(timeoutId);
  }
};

// ⭐ REVAMPED PDF GENERATION - Now handles bold text, patient details, and embeds the plot image ⭐
const generatePdfFromMarkdown = async (markdown: string, data: ReportRequest): Promise<string> => {
  const { patientDetails, shap_analysis } = data;
  const doc = new jsPDF();
  const margin = 15;
  const pageWidth = doc.internal.pageSize.getWidth();
  const usableWidth = pageWidth - 2 * margin;
  let yPosition = margin;

  // --- Add Patient Details Header ---
  doc.setFont("helvetica", "bold");
  doc.setFontSize(18);
  doc.text("Gallstone Risk Analysis Report", pageWidth / 2, yPosition, { align: "center" });
  yPosition += 10;
  
  doc.setFont("helvetica", "normal");
  doc.setFontSize(12);
  doc.text(`Patient Name: ${patientDetails.name}`, margin, yPosition);
  yPosition += 7;
  doc.text(`Address: ${patientDetails.address}`, margin, yPosition);
  yPosition += 7;
  doc.text(`Phone: ${patientDetails.phone}`, margin, yPosition);
  yPosition += 10;
  doc.line(margin, yPosition, pageWidth - margin, yPosition); // Horizontal line
  yPosition += 10;

  // --- Function to add text and handle page breaks ---
  const addText = (text: string, options: { isBold?: boolean } = {}) => {
    doc.setFont("helvetica", options.isBold ? "bold" : "normal");
    const lines = doc.splitTextToSize(text, usableWidth);
    const textHeight = doc.getTextDimensions(lines).h;
    if (yPosition + textHeight > doc.internal.pageSize.getHeight() - margin) {
      doc.addPage();
      yPosition = margin;
    }
    doc.text(lines, margin, yPosition);
    yPosition += textHeight + 4; // Spacing
  };

  // --- Parse Markdown and Add to PDF ---
  const mdLines = markdown.split('\n');
  for (const line of mdLines) {
    if (line.startsWith('## ')) {
      doc.setFont("helvetica", "bold");
      doc.setFontSize(16);
      addText(line.replace('## ', ''));
    } else if (line.startsWith('# ')) {
      doc.setFont("helvetica", "bold");
      doc.setFontSize(20);
      addText(line.replace('# ', ''));
    } else if (line.trim().startsWith('* ')) {
      doc.setFont("helvetica", "normal");
      doc.setFontSize(12);
      // Handle bold within list items
      const itemText = line.trim().substring(2).replace(/\*\*(.*?)\*\*/g, '$1'); // Simplified for now
      addText(`• ${itemText}`);
    } else if (line.trim().length > 0) {
      // FIX: Handle **bold** text within paragraphs
      doc.setFont("helvetica", "normal");
      doc.setFontSize(12);
      const parts = line.split(/\*\*(.*?)\*\*/g); // Split by bold tags
      let currentX = margin;
      
      if (yPosition + 10 > doc.internal.pageSize.getHeight() - margin) {
          doc.addPage();
          yPosition = margin;
      }
      
      parts.forEach((part, index) => {
        if (part) {
          const isBold = index % 2 === 1;
          doc.setFont("helvetica", isBold ? "bold" : "normal");
          doc.text(part, currentX, yPosition);
          currentX += doc.getStringUnitWidth(part) * doc.getFontSize();
        }
      });
      yPosition += 7; // Move to next line
    }
  }

  // --- Fetch and Embed SHAP Plot Image ---
  try {
    yPosition += 5;
    doc.setFont("helvetica", "bold");
    doc.setFontSize(16);
    addText("SHAP Waterfall Plot");
    
    const imageUrl = shap_analysis.plots.waterfall_plot;
    const imageResponse = await fetch(imageUrl);
    const imageBuffer = await imageResponse.arrayBuffer();
    const imageBase64 = Buffer.from(imageBuffer).toString('base64');
    
    // Assuming the plot is PNG, you might need to adjust if it's JPG
    const imageFormat = 'PNG'; 
    const imgProps = doc.getImageProperties(imageBase64);
    const imgWidth = usableWidth;
    const imgHeight = (imgProps.height * imgWidth) / imgProps.width;

    if (yPosition + imgHeight > doc.internal.pageSize.getHeight() - margin) {
      doc.addPage();
      yPosition = margin;
    }
    
    doc.addImage(imageBase64, imageFormat, margin, yPosition, imgWidth, imgHeight);

  } catch(e) {
      console.error("Failed to embed plot image:", e);
      addText("Error: The SHAP plot image could not be loaded and embedded.", { isBold: true });
  }

  return Buffer.from(doc.output('arraybuffer')).toString("base64");
};

// =================================================================================
// 3. BACKGROUND JOB PROCESSOR
// =================================================================================

async function processReport(jobId: string, data: ReportRequest) {
  try {
    const markdownContent = await generateMarkdownReport(data);
    // Pass the full data object to the PDF generator
    const pdfBase64 = await generatePdfFromMarkdown(markdownContent, data);
    jobStore.set(jobId, { status: "complete", pdf: pdfBase64 });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : "An unknown error occurred.";
    jobStore.set(jobId, { status: "failed", error: errorMessage });
  }
}

// =================================================================================
// 4. API ROUTE HANDLERS
// =================================================================================

export async function POST(req: NextRequest) {
  try {
    const data: ReportRequest = await req.json();
    const jobId = randomUUID();
    jobStore.set(jobId, { status: "pending" });
    processReport(jobId, data);
    return NextResponse.json({ jobId }, { status: 202 });
  } catch (error) {
    return NextResponse.json({ error: "Invalid request body." }, { status: 400 });
  }
}

export async function GET(req: NextRequest) {
  const jobId = req.nextUrl.searchParams.get("jobId");
  if (!jobId) {
    return NextResponse.json({ error: "jobId query parameter is required." }, { status: 400 });
  }
  const job = jobStore.get(jobId);
  if (!job) {
    return NextResponse.json({ error: "Job not found." }, { status: 404 });
  }
  if (job.status === "complete" || job.status === "failed") {
    jobStore.delete(jobId);
  }
  return NextResponse.json(job);
}