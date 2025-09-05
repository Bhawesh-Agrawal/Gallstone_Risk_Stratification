import { NextRequest, NextResponse } from "next/server";
import { jsPDF } from "jspdf";
import { randomUUID } from "crypto";

// =================================================================================
// 1. IN-MEMORY JOB STORE & TYPES
// =================================================================================

// MODIFIED: Added formData to the request interface
interface ReportRequest {
  patientDetails: {
    name: string;
    address: string;
    phone: string;
  };
  formData: Record<string, number>; // All the medical form data
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

// NEW: Copied from frontend to structure the input data section in the PDF
const fieldCategories = {
  "Basic Information": ["Age", "Gender", "Height", "Weight"],
  "Medical History": ["Comorbidity", "Coronary Artery Disease (CAD)", "Hypothyroidism", "Hyperlipidemia", "Diabetes Mellitus (DM)", "Hepatic Fat Accumulation (HFA)"],
  "Body Composition": ["Body Mass Index (BMI)", "Total Body Water (TBW)", "Extracellular Water (ECW)", "Intracellular Water (ICW)", "Extracellular Fluid/Total Body Water (ECF/TBW)", "Total Body Fat Ratio (TBFR) (%)", "Lean Mass (LM) (%)", "Body Protein Content (Protein) (%)", "Visceral Fat Rating (VFR)", "Bone Mass (BM)", "Muscle Mass (MM)", "Obesity (%)", "Total Fat Content (TFC)", "Visceral Fat Area (VFA)", "Visceral Muscle Area (VMA) (Kg)"],
  "Laboratory Tests": ["Glucose", "Total Cholesterol (TC)", "Low Density Lipoprotein (LDL)", "High Density Lipoprotein (HDL)", "Triglyceride", "Aspartat Aminotransferaz (AST)", "Alanin Aminotransferaz (ALT)", "Alkaline Phosphatase (ALP)", "Creatinine", "Glomerular Filtration Rate (GFR)", "C-Reactive Protein (CRP)", "Hemoglobin (HGB)", "Vitamin D"],
};


// =================================================================================
// 2. CORE LOGIC (Gemini API Call and PDF Generation)
// =================================================================================

const generateMarkdownReport = async (data: ReportRequest): Promise<string> => {
  const { prediction_label, probability, shap_analysis } = data;

  // MODIFIED: Slightly adjusted prompt for more structured Markdown output.
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
    1.  Start with a title: "# Gallstone Risk Analysis Report".
    2.  Create a section "## Prediction Summary". Summarize the prediction outcome and probabilities.
    3.  Create a section "## Understanding SHAP Values". Explain SHAP in simple terms.
    4.  Create a section "## Factors Increasing Gallstone Risk". Discuss the "Top contributors to positive" and explain the medical reasons why they might increase the risk.
    5.  Create a section "## Factors Decreasing Gallstone Risk". Discuss the "Top contributors to negative" and explain why they might be protective.
    6.  Conclude with a final section "## Disclaimer" stating this is not medical advice.
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

// ⭐ COMPLETELY REVAMPED PDF GENERATION FUNCTION ⭐
const generatePdfFromMarkdown = async (markdown: string, data: ReportRequest): Promise<string> => {
    const { patientDetails, shap_analysis, formData, prediction_label } = data;
    const doc = new jsPDF();
    
    // --- Document Setup ---
    const margin = 15;
    const pageHeight = doc.internal.pageSize.getHeight();
    const pageWidth = doc.internal.pageSize.getWidth();
    const usableWidth = pageWidth - 2 * margin;
    let yPosition = margin;
    let pageNumber = 1;

    // --- Helper Functions ---
    const checkPageBreak = (heightNeeded: number = 10) => {
        if (yPosition + heightNeeded > pageHeight - margin) {
            addFooter();
            doc.addPage();
            pageNumber++;
            yPosition = margin;
        }
    };
    
    const addFooter = () => {
        const footerY = pageHeight - 10;
        doc.setFontSize(8);
        doc.setTextColor(150);
        doc.text(
            `Disclaimer: This report is for informational purposes only. Consult a healthcare professional for medical advice.`,
            margin,
            footerY,
            { maxWidth: usableWidth }
        );
        doc.text(`Page ${pageNumber}`, pageWidth - margin, footerY, { align: "right" });
    };

    const addSectionTitle = (title: string) => {
        checkPageBreak(20);
        yPosition += 10;
        doc.setFont("helvetica", "bold");
        doc.setFontSize(14);
        doc.setTextColor(40, 52, 140); // Dark blue color
        doc.text(title, margin, yPosition);
        yPosition += 2;
        doc.setDrawColor(40, 52, 140);
        doc.line(margin, yPosition, margin + usableWidth, yPosition);
        yPosition += 10;
    };

    const addStyledText = (text: string) => {
        doc.setFont("helvetica", "normal");
        doc.setFontSize(10);
        doc.setTextColor(80);

        const lines = doc.splitTextToSize(text.replace(/•\s/g, ''), usableWidth - (text.startsWith('•') ? 5 : 0));
        
        lines.forEach((line: string, index: number) => {
            checkPageBreak(5);
            const xPos = margin + (text.startsWith('•') ? 5 : 0);
            if (index === 0 && text.startsWith('•')) {
                 doc.setFont("helvetica", "bold");
                 doc.text("•", margin, yPosition);
                 doc.setFont("helvetica", "normal");
            }
            doc.text(line, xPos, yPosition);
            yPosition += 5;
        });
        yPosition += 3; // Extra space after paragraph/item
    };
    
    // --- 1. Report Header ---
    doc.setFont("helvetica", "bold");
    doc.setFontSize(18);
    doc.setTextColor(0);
    doc.text("Confidential Medical Report", pageWidth / 2, yPosition, { align: "center" });
    yPosition += 8;
    doc.setFontSize(14);
    doc.setTextColor(100);
    doc.text("Gallstone Risk Prediction Analysis", pageWidth / 2, yPosition, { align: "center" });
    yPosition += 12;

    doc.setDrawColor(200);
    doc.line(margin, yPosition, pageWidth - margin, yPosition);
    yPosition += 10;

    doc.setFont("helvetica", "bold");
    doc.setFontSize(10);
    doc.text("Patient Name:", margin, yPosition);
    doc.setFont("helvetica", "normal");
    doc.text(patientDetails.name, margin + 40, yPosition);
    yPosition += 6;
    doc.setFont("helvetica", "bold");
    doc.text("Address:", margin, yPosition);
    doc.setFont("helvetica", "normal");
    doc.text(patientDetails.address, margin + 40, yPosition);
    yPosition += 6;
    doc.setFont("helvetica", "bold");
    doc.text("Phone:", margin, yPosition);
    doc.setFont("helvetica", "normal");
    doc.text(patientDetails.phone, margin + 40, yPosition);
    yPosition += 6;
    doc.setFont("helvetica", "bold");
    doc.text("Report Date:", margin, yPosition);
    doc.setFont("helvetica", "normal");
    doc.text(new Date().toLocaleDateString('en-GB'), margin + 40, yPosition);

    // --- 2. Prediction Summary ---
    addSectionTitle("Prediction Summary");
    const isPositive = prediction_label === 'Positive';
    const summaryColor = isPositive ? [217, 48, 30] : [22, 163, 74];
    const summaryBg = isPositive ? [254, 226, 226] : [220, 252, 231];
    doc.setFillColor(summaryBg[0], summaryBg[1], summaryBg[2]);
    doc.rect(margin, yPosition, usableWidth, 20, 'F');
    doc.setFont("helvetica", "bold");
    doc.setFontSize(12);
    doc.setTextColor(summaryColor[0], summaryColor[1], summaryColor[2]);
    doc.text(`Prediction: ${prediction_label} for Gallstones`, pageWidth / 2, yPosition + 12, { align: "center" });
    yPosition += 25;

    // --- 3. Input Data Summary (NEW SECTION) ---
    addSectionTitle("Patient Data Summary");
    doc.setFontSize(10);
    const columnWidth = usableWidth / 3;
    let currentX = margin;
    let startY = yPosition;
    let maxY = yPosition;

    Object.entries(fieldCategories).forEach(([category, fields]) => {
        checkPageBreak(8 + fields.length * 5);
        yPosition += 3;
        doc.setFont("helvetica", "bold");
        doc.setTextColor(40, 52, 140);
        doc.text(category, currentX, yPosition);
        yPosition += 6;

        (fields as string[]).forEach((field) => {
            checkPageBreak(5);
            const value = formData[field as keyof typeof formData];
            let displayValue = "N/A";
            if (value !== undefined) {
                if (["Gender", "Comorbidity", "Coronary Artery Disease (CAD)", "Hypothyroidism", "Hyperlipidemia", "Diabetes Mellitus (DM)", "Hepatic Fat Accumulation (HFA)"].includes(field)){
                     displayValue = value === 1 ? 'Yes' : 'No';
                     if (field === 'Gender') displayValue = value === 1 ? 'Female' : 'Male';
                } else {
                     displayValue = String(value);
                }
            }
            doc.setFont("helvetica", "bold");
            doc.setTextColor(80);
            doc.text(`${field}:`, currentX, yPosition);
            doc.setFont("helvetica", "normal");
            doc.setTextColor(0);
            doc.text(displayValue, currentX + 60, yPosition);
            yPosition += 5;
        });
    });

    // --- 4. Detailed AI Analysis ---
    const markdownSections = markdown.split('## ').filter(s => s.trim());
    markdownSections.forEach(section => {
        const lines = section.split('\n').filter(l => l.trim());
        const title = lines.shift() || 'Details';
        const content = lines.join('\n').replace(/\*\*/g, ''); // Simple bold removal for now
        addSectionTitle(title);
        content.split('\n').forEach(paragraph => {
            if (paragraph.trim()) {
                 addStyledText(paragraph.trim());
            }
        });
    });

    // --- 5. SHAP Waterfall Plot ---
    try {
        addSectionTitle("SHAP Waterfall Plot");
        addStyledText("This plot shows the specific impact of each top feature on this patient's risk prediction. Features in red increase the predicted risk, while those in blue decrease it.");
        
        const imageUrl = shap_analysis.plots.waterfall_plot;
        const imageResponse = await fetch(imageUrl);
        const imageBuffer = await imageResponse.arrayBuffer();
        const imageBase64 = Buffer.from(imageBuffer).toString('base64');
        const imageFormat = 'PNG';
        const imgProps = doc.getImageProperties(imageBase64);
        
        const imgWidth = usableWidth;
        const imgHeight = (imgProps.height * imgWidth) / imgProps.width;

        checkPageBreak(imgHeight + 10);
        doc.addImage(imageBase64, imageFormat, margin, yPosition, imgWidth, imgHeight);

    } catch (e) {
        console.error("Failed to embed plot image:", e);
        addStyledText("Error: The SHAP plot image could not be loaded and embedded.");
    }
    
    // --- Final Footer ---
    addFooter();

    return Buffer.from(doc.output('arraybuffer')).toString("base64");
};


// =================================================================================
// 3. BACKGROUND JOB PROCESSOR
// =================================================================================

async function processReport(jobId: string, data: ReportRequest) {
  try {
    const markdownContent = await generateMarkdownReport(data);
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
    // Basic validation
    if (!data.patientDetails || !data.formData || !data.prediction_label) {
        throw new Error("Invalid request body. Missing required fields.");
    }
    const jobId = randomUUID();
    jobStore.set(jobId, { status: "pending" });
    processReport(jobId, data);
    return NextResponse.json({ jobId }, { status: 202 });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Invalid request body.";
    return NextResponse.json({ error: message }, { status: 400 });
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