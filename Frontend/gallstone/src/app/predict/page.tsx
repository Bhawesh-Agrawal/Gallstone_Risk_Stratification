"use client";

import React, { useEffect, useMemo, useState, useCallback, useRef } from "react";
// --- Convex Imports (NEW) ---
import { useAction } from "convex/react";
import { api } from "../../../convex/_generated/api";
// ----------------------------
import type { LucideIcon } from "lucide-react";
import {
  AlertCircle,
  Activity,
  User,
  TestTube,
  Heart,
  Scale,
  Info,
  CheckCircle,
  XCircle,
  Loader2,
  FileText,
  Download,
} from "lucide-react";
import { toast } from "sonner";
import { Id } from "../../../convex/_generated/dataModel"

// -------------------- Field Definitions & Types --------------------
const fieldDescriptions = {
  Age: "The patient's age in years. Example: 45. Use whole or decimal values.",
  Gender: "Patient's biological sex: 0 = Male, 1 = Female. Select from the dropdown.",
  Comorbidity: "Indicates whether the patient has other diseases (e.g., hypertension, asthma). 0 = No, 1 = Yes.",
  "Coronary Artery Disease (CAD)": "Heart disease caused by narrowing or blockage of coronary arteries. 0 = No, 1 = Yes.",
  Hypothyroidism: "Thyroid hormone deficiency causing slow metabolism. 0 = No, 1 = Yes.",
  Hyperlipidemia: "Elevated lipid levels in blood, increasing heart disease risk. 0 = No, 1 = Yes.",
  "Diabetes Mellitus (DM)": "Metabolic disorder with high blood sugar. 0 = No, 1 = Yes.",
  Height: "Height in centimeters. Example: 170 cm.",
  Weight: "Weight in kilograms. Example: 70.5 kg.",
  "Body Mass Index (BMI)": "BMI = Weight (kg) / [Height (m)]². Normal: 18.5–24.9, Overweight: 25–29.9, Obese: 30+.",
  "Total Body Water (TBW)": "Percentage of body weight made up of water. Normal: 50–65%.",
  "Extracellular Water (ECW)": "Water outside the cells (plasma, interstitial fluid). Typical: ~20% of weight.",
  "Intracellular Water (ICW)": "Water inside cells. Usually ~40% of body weight.",
  "Extracellular Fluid/Total Body Water (ECF/TBW)": "Ratio to check fluid balance. High values suggest fluid retention or inflammation.",
  "Total Body Fat Ratio (TBFR) (%)": "Percentage of body weight that is fat. Healthy range: Males 8–19%, Females 21–33%.",
  "Lean Mass (LM) (%)": "Percentage of muscle, bone, organs. High lean mass = healthier composition.",
  "Body Protein Content (Protein) (%)": "Protein percentage in lean mass. Indicates muscle and organ function.",
  "Visceral Fat Rating (VFR)": "Index for fat stored around internal organs. High levels = metabolic risk.",
  "Bone Mass (BM)": "Estimated bone mineral weight in kg. Normal range: ~3–4 kg depending on body size.",
  "Muscle Mass (MM)": "Muscle tissue weight in kg.",
  "Obesity (%)": "Overall fat percentage classification. >25% (men) or >32% (women) = obese.",
  "Total Fat Content (TFC)": "Absolute mass of fat in the body (kg). Example: 15–30 kg.",
  "Visceral Fat Area (VFA)": "Cross-sectional abdominal fat area (cm²). >100 cm² = high risk.",
  "Visceral Muscle Area (VMA) (Kg)": "Muscle mass in the abdominal region (kg). Important for core strength.",
  "Hepatic Fat Accumulation (HFA)": "Fat buildup in the liver (fatty liver disease). 0 = No, 1 = Yes.",
  Glucose: "Blood sugar level (mg/dL). Normal fasting: 70–99 mg/dL. Diabetes: 126+.",
  "Total Cholesterol (TC)": "Sum of all cholesterol. Normal < 200 mg/dL. High ≥ 240.",
  "Low Density Lipoprotein (LDL)": "Bad cholesterol. Optimal < 100 mg/dL. High ≥ 160.",
  "High Density Lipoprotein (HDL)": "Good cholesterol. Low < 40 (men) or < 50 (women). High ≥ 60 = protective.",
  Triglyceride: "Fats in blood used for energy. Normal < 150 mg/dL. High ≥ 200.",
  "Aspartat Aminotransferaz (AST)": "Liver enzyme. Normal: 10–40 U/L. High = liver/muscle damage.",
  "Alanin Aminotransferaz (ALT)": "Liver enzyme. Normal: 7–56 U/L. High = liver inflammation.",
  "Alkaline Phosphatase (ALP)": "Enzyme linked to liver/bone. Normal: 44–147 U/L. High = blockage/bone disorder.",
  Creatinine: "Kidney function marker. Normal: 0.6–1.2 mg/dL. High = kidney dysfunction.",
  "Glomerular Filtration Rate (GFR)": "Kidney filtering efficiency. Normal ≥ 90 mL/min/1.73m². Low = kidney disease.",
  "C-Reactive Protein (CRP)": "Inflammation marker. Normal < 1 mg/L. High = infection/inflammation.",
  "Hemoglobin (HGB)": "Oxygen-carrying protein. Normal: Men 13.8–17.2 g/dL, Women 12.1–15.1.",
  "Vitamin D": "Vitamin D blood level (ng/mL). Deficiency < 20, sufficient 30–50.",
} as const;

export type FieldName = keyof typeof fieldDescriptions;
export type FormData = Partial<Record<FieldName, number>>;
type PatientDetails = { name: string; address: string; phone: string };

type PredictionSuccess = {
  prediction_label: "Positive" | "Negative";
  probability: { "positive (class 0)": string; "negative (class 1)": string; };
  shap_analysis: {
    base_value_for_positive_class: number;
    base_value_for_negative_class: number;
    top_contributors_to_positive: Record<string, string>;
    top_contributors_to_negative: Record<string, string>;
    plots: { waterfall_plot: string; bar_summary_plot: string; };
  };
};

type PredictionError = { error: string };
type ApiResponse = PredictionSuccess | PredictionError;

type ReportJobStatus = {
  status: "pending" | "complete" | "failed";
  pdf?: string;
  error?: string;
};

// -------------------- Field Configuration --------------------
const dropdownFields: Readonly<FieldName[]> = ["Gender", "Comorbidity", "Coronary Artery Disease (CAD)", "Hypothyroidism", "Hyperlipidemia", "Diabetes Mellitus (DM)", "Hepatic Fat Accumulation (HFA)"];

const fieldCategories = {
  "Patient Details": ["name", "address", "phone"],
  "Basic Information": ["Age", "Gender", "Height", "Weight"],
  "Medical History": ["Comorbidity", "Coronary Artery Disease (CAD)", "Hypothyroidism", "Hyperlipidemia", "Diabetes Mellitus (DM)", "Hepatic Fat Accumulation (HFA)"],
  "Body Composition": ["Body Mass Index (BMI)", "Total Body Water (TBW)", "Extracellular Water (ECW)", "Intracellular Water (ICW)", "Extracellular Fluid/Total Body Water (ECF/TBW)", "Total Body Fat Ratio (TBFR) (%)", "Lean Mass (LM) (%)", "Body Protein Content (Protein) (%)", "Visceral Fat Rating (VFR)", "Bone Mass (BM)", "Muscle Mass (MM)", "Obesity (%)", "Total Fat Content (TFC)", "Visceral Fat Area (VFA)", "Visceral Muscle Area (VMA) (Kg)"],
  "Laboratory Tests": ["Glucose", "Total Cholesterol (TC)", "Low Density Lipoprotein (LDL)", "High Density Lipoprotein (HDL)", "Triglyceride", "Aspartat Aminotransferaz (AST)", "Alanin Aminotransferaz (ALT)", "Alkaline Phosphatase (ALP)", "Creatinine", "Glomerular Filtration Rate (GFR)", "C-Reactive Protein (CRP)", "Hemoglobin (HGB)", "Vitamin D"],
} as const;

export type CategoryName = keyof typeof fieldCategories;
const categoryIcons: Record<CategoryName, LucideIcon> = {
  "Patient Details": User,
  "Basic Information": User,
  "Medical History": Heart,
  "Body Composition": Scale,
  "Laboratory Tests": TestTube,
};
const categoryList = Object.keys(fieldCategories) as CategoryName[];

// -------------------- Reusable Input Component --------------------
type FormFieldProps = {
  field: FieldName;
  value: number | undefined;
  onChange: (field: FieldName, value: number | undefined) => void;
  isBMICalculated: boolean;
};
const FormField: React.FC<FormFieldProps> = ({ field, value, onChange, isBMICalculated }) => {
  const [showTooltip, setShowTooltip] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const rawValue = e.target.value;
    onChange(field, rawValue === "" ? undefined : parseFloat(rawValue));
  };

  const dropdownOptions =
    field === "Gender" ? [{ value: 0, label: "Male" }, { value: 1, label: "Female" }] : [{ value: 0, label: "No" }, { value: 1, label: "Yes" }];

  return (
    <div className="relative">
      <label className="block text-sm font-semibold text-gray-800 mb-2">
        <span className="flex items-center gap-2">
          {field}
          {value !== undefined && <CheckCircle className="h-4 w-4 text-green-600" />}
          <button type="button" aria-label={`Info about ${field}`} onMouseEnter={() => setShowTooltip(true)} onMouseLeave={() => setShowTooltip(false)}>
            <Info className="h-4 w-4 text-blue-600" />
          </button>
        </span>
      </label>
      {showTooltip && (
        <div className="absolute z-10 w-80 p-3 text-sm text-white bg-gray-800 rounded-lg shadow-lg -top-2 left-0 transform -translate-y-full">
          {fieldDescriptions[field]}
          <div className="absolute top-full left-4 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-800" />
        </div>
      )}
      {dropdownFields.includes(field) ? (
        <select value={value !== undefined ? String(value) : ""} onChange={handleChange} required className="w-full p-3 border border-gray-300 rounded-lg">
          <option value="">Select...</option>
          {dropdownOptions.map((opt) => (<option key={opt.value} value={opt.value}> {opt.label} </option>))}
        </select>
      ) : (
        <input type="number" step="any" value={value !== undefined ? String(value) : ""} onChange={handleChange} required className="w-full p-3 border border-gray-300 rounded-lg" min={0} placeholder={`Enter ${field}`} readOnly={field === "Body Mass Index (BMI)" && isBMICalculated} />
      )}
    </div>
  );
};

// -------------------- Main Predict Component --------------------
export default function Predict(): React.ReactElement {
  const [patientDetails, setPatientDetails] = useState<PatientDetails>({ name: "", address: "", phone: "" });
  const [formData, setFormData] = useState<FormData>({});
  const [response, setResponse] = useState<ApiResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [currentCategory, setCurrentCategory] = useState<CategoryName>("Patient Details");

  const [isReportLoading, setIsReportLoading] = useState<boolean>(false);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const pollingRef = useRef<number | null>(null);

  // --- Convex Action Hook (NEW) ---
  const saveReport = useAction(api.report.saveReport);
  // ---------------------------------

  const totalFields = useMemo(() => Object.keys(fieldDescriptions).length, []);
  const completedFields = useMemo(() => Object.values(formData).filter((v) => v !== undefined).length, [formData]);
  const progressPercentage = useMemo(() => (totalFields > 0 ? (completedFields / totalFields) * 100 : 0), [completedFields, totalFields]);

  const handleFieldChange = useCallback((field: FieldName, value: number | undefined) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  }, []);

  useEffect(() => {
    const height = formData.Height;
    const weight = formData.Weight;
    if (height && weight && height > 0) {
      const heightInMeters = height / 100;
      const bmi = weight / (heightInMeters * heightInMeters);
      handleFieldChange("Body Mass Index (BMI)", parseFloat(bmi.toFixed(2)));
    }
  }, [formData.Height, formData.Weight, handleFieldChange]);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (completedFields < totalFields) {
      toast.error(`Please complete all ${totalFields} medical fields.`);
      return;
    }
    setIsLoading(true);
    setResponse(null);
    setPdfUrl(null);
    setJobId(null);
    if (pollingRef.current) clearInterval(pollingRef.current);

    try {
      const res = await fetch("https://codexbhawesh-gallstone.hf.space/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });
      if (!res.ok) throw new Error(`API request failed with status: ${res.status}`);
      const data: ApiResponse = await res.json();
      setResponse(data);
      if ("error" in data) {
        toast.error(`Prediction Failed: ${data.error}`);
      } else {
        toast.success("Prediction successful!");
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "An unknown error occurred.";
      setResponse({ error: errorMessage });
      toast.error(`Prediction Failed: ${errorMessage}`);
    } finally {
      setIsLoading(false);
    }
  };

  const base64ToBlob = (base64: string, mimeType: string) => {
    const byteCharacters = atob(base64);
    const byteArrays = [];
    for (let offset = 0; offset < byteCharacters.length; offset += 512) {
      const slice = byteCharacters.slice(offset, offset + 512);
      const byteNumbers = new Array(slice.length);
      for (let i = 0; i < slice.length; i++) {
        byteNumbers[i] = slice.charCodeAt(i);
      }
      byteArrays.push(new Uint8Array(byteNumbers));
    }
    return new Blob(byteArrays, { type: mimeType });
  };

  const handleGenerateReport = async () => {
    if (!response || "error" in response) {
      toast.error("A successful prediction is required to generate a report.");
      return;
    }
    if (Object.values(patientDetails).some(v => v === "")) {
      toast.error("Please fill in all patient details before generating a report.");
      setCurrentCategory("Patient Details");
      return;
    }

    setIsReportLoading(true);
    setPdfUrl(null);
    toast.info("Report generation started in the background...");

    try {
      const payload = { ...response, patientDetails };
      const res = await fetch("/api/generate-report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (res.status !== 202) {
        const errorData = await res.json();
        throw new Error(errorData.error || "Failed to start report job.");
      }
      const { jobId } = await res.json();
      setJobId(jobId);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "An unknown error occurred.";
      toast.error(`Failed to start report: ${errorMessage}`);
      setIsReportLoading(false);
    }
  };

  useEffect(() => {
    if (!jobId) return;

    let pollCount = 0;
    const maxPolls = 40;

    const checkStatus = async () => {
      if (pollCount >= maxPolls) {
        toast.error("Report generation timed out.");
        if (pollingRef.current) clearInterval(pollingRef.current);
        setIsReportLoading(false);
        setJobId(null);
        return;
      }

      try {
        const res = await fetch(`/api/generate-report?jobId=${jobId}`);
        const data: ReportJobStatus = await res.json();

        if (data.status === "complete") {
          if (pollingRef.current) clearInterval(pollingRef.current);
          if (data.pdf) {
            const pdfBlob = base64ToBlob(data.pdf, "application/pdf");
            setPdfUrl(URL.createObjectURL(pdfBlob));
            toast.success("Report generated successfully!");

            // --- Save to Convex Database (NEW LOGIC) ---
            const userId = localStorage.getItem("userId");
            if (!userId) {
              toast.error("Could not find userId in localStorage. Report not saved.");
            } else if (response && "prediction_label" in response) {
              try {
                // Call the Convex action to save everything
                await saveReport({
                  userId: userId as Id<"users">,
                  patientDetails,
                  formData,
                  predictionResult: response,
                  pdfBase64: data.pdf, // Pass the raw base64 string
                });
                toast.success("Report details saved to the database!");
              } catch (error) {
                console.error("Failed to save report to Convex:", error);
                toast.error("Could not save report to the database.");
              }
            }
            // ------------------------------------------
          }
          setIsReportLoading(false);
          setJobId(null);
        } else if (data.status === "failed") {
          if (pollingRef.current) clearInterval(pollingRef.current);
          toast.error(`Report generation failed: ${data.error}`);
          setIsReportLoading(false);
          setJobId(null);
        }
      } catch (err) {
        if (pollingRef.current) clearInterval(pollingRef.current);
        toast.error("Error checking report status.");
        setIsReportLoading(false);
        setJobId(null);
      }
      pollCount++;
    };

    pollingRef.current = window.setInterval(checkStatus, 3000);
    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, [jobId, response, patientDetails, formData, saveReport]); // Added dependencies

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4 flex items-center justify-center gap-3">
            <Activity className="h-10 w-10 text-blue-600" /> Gallstone Risk Prediction
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Complete the form with your health details for a gallstone risk assessment.
          </p>
          <div className="mt-6 max-w-md mx-auto">
            <div className="flex justify-between text-sm text-gray-600 mb-2">
              <span>Progress</span>
              <span>{completedFields}/{totalFields} fields</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3">
              <div className="bg-gradient-to-r from-blue-500 to-blue-600 h-3 rounded-full" style={{ width: `${progressPercentage}%` }} />
            </div>
          </div>
        </div>

        {/* Disclaimer */}
        <div className="bg-amber-50 border-l-4 border-amber-400 p-4 mb-8 rounded-r-lg">
          <div className="flex items-center">
            <AlertCircle className="h-5 w-5 text-amber-600 mr-3" />
            <p className="text-amber-800 font-semibold">
              Medical Disclaimer: This tool is for educational purposes only and not a substitute for professional medical advice.
            </p>
          </div>
        </div>

        <div className="grid lg:grid-cols-4 gap-8">
          {/* Navigation */}
          <aside className="lg-col-span-1">
            <div className="bg-white rounded-xl shadow-lg p-6 lg:sticky lg:top-8">
              <h3 className="font-semibold text-gray-900 mb-4">Form Sections</h3>
              <nav className="space-y-2">
                {categoryList.map((category) => {
                  const Icon = categoryIcons[category];
                  const categoryFields = fieldCategories[category];
                  const isPatientSection = category === "Patient Details";
                  const completedCount = isPatientSection
                    ? Object.values(patientDetails).filter(Boolean).length
                    : categoryFields.filter(field => formData[field as FieldName] !== undefined).length;
                  const totalCount = categoryFields.length;
                  const isComplete = completedCount === totalCount;
                  return (
                    <button key={category} type="button" onClick={() => setCurrentCategory(category)} className={`w-full flex items-center gap-3 p-3 rounded-lg text-left transition-all ${currentCategory === category ? "bg-blue-100 text-blue-800 border-2 border-blue-300" : "hover:bg-gray-50 text-gray-700"}`}>
                      <Icon className="h-5 w-5" />
                      <div className="flex-1">
                        <div className="font-medium">{category}</div>
                        <div className="text-sm text-gray-500">{`${completedCount}/${totalCount}`}</div>
                      </div>
                      {isComplete && <CheckCircle className="h-5 w-5 text-green-600" />}
                    </button>
                  );
                })}
              </nav>
            </div>
          </aside>

          {/* Main Form & Results */}
          <main className="lg:col-span-3">
            <form onSubmit={handleSubmit} className="space-y-8" noValidate>
              {categoryList.map(category => {
                const Icon = categoryIcons[category];
                return currentCategory === category && (
                  <section key={category} className="bg-white rounded-xl shadow-lg p-8 ring-2 ring-blue-500">
                    <div className="flex items-center gap-3 mb-6">
                      <Icon className="h-6 w-6 text-blue-600" />
                      <h2 className="text-2xl font-bold text-gray-900">{category}</h2>
                    </div>
                    {category === "Patient Details" ? (
                      <div className="grid md:grid-cols-2 gap-6">
                        <div>
                          <label className="block text-sm font-semibold text-gray-800 mb-2">Name</label>
                          <input type="text" value={patientDetails.name} onChange={(e) => setPatientDetails(p => ({ ...p, name: e.target.value }))} required className="w-full p-3 border border-gray-300 rounded-lg" />
                        </div>
                        <div>
                          <label className="block text-sm font-semibold text-gray-800 mb-2">Address</label>
                          <input type="text" value={patientDetails.address} onChange={(e) => setPatientDetails(p => ({ ...p, address: e.target.value }))} required className="w-full p-3 border border-gray-300 rounded-lg" />
                        </div>
                        <div>
                          <label className="block text-sm font-semibold text-gray-800 mb-2">Phone Number</label>
                          <input type="tel" value={patientDetails.phone} onChange={(e) => setPatientDetails(p => ({ ...p, phone: e.target.value }))} required className="w-full p-3 border border-gray-300 rounded-lg" />
                        </div>
                      </div>
                    ) : (
                      <div className="grid md:grid-cols-2 gap-6">
                        {(fieldCategories[category] as readonly FieldName[]).map((field) => (
                          <FormField key={field} field={field} value={formData[field]} onChange={handleFieldChange} isBMICalculated={!!(formData.Height && formData.Weight)} />
                        ))}
                      </div>
                    )}
                  </section>
                )
              })}
              <div className="text-center">
                <button type="submit" disabled={isLoading || completedFields < totalFields} className="bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 disabled:from-gray-400 disabled:to-gray-500 text-white font-bold py-4 px-8 rounded-xl text-lg transition-all transform hover:scale-105 disabled:scale-100 disabled:cursor-not-allowed shadow-lg">
                  {isLoading ? <span className="flex items-center gap-2 justify-center"> <Loader2 className="h-5 w-5 animate-spin" /> Processing... </span> : `Get Prediction`}
                </button>
              </div>
            </form>
            {response && (
              <div className="mt-8 bg-white rounded-xl shadow-lg p-8">
                <h3 className="text-2xl font-bold text-gray-900 mb-4 flex items-center gap-3">
                  {"error" in response ? <><XCircle className="h-6 w-6 text-red-600" /> Prediction Error</> : <><CheckCircle className="h-6 w-6 text-green-600" /> Prediction Result</>}
                </h3>
                {"error" in response ? (<div className="bg-red-50 border border-red-200 rounded-lg p-4"> <p className="text-red-800">{response.error}</p> </div>) : (
                  <div className="space-y-6">
                    <div className={`p-6 rounded-lg text-center ${response.prediction_label === "Positive" ? "bg-red-100 border-red-300" : "bg-green-100 border-green-300"} border-2`}>
                      <p className="text-lg font-medium text-gray-700 mb-2">Prediction Outcome</p>
                      <p className={`text-4xl font-bold ${response.prediction_label === "Positive" ? "text-red-700" : "text-green-700"}`}> {`${response.prediction_label} for Gallstones`} </p>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-center">
                      <div className="p-4 bg-gray-50 rounded-lg border">
                        <p className="text-sm font-medium text-gray-600">Probability (Positive)</p>
                        <p className="text-2xl font-semibold text-gray-800"> {(parseFloat(response.probability["positive (class 0)"]) * 100).toFixed(2)}% </p>
                      </div>
                      <div className="p-4 bg-gray-50 rounded-lg border">
                        <p className="text-sm font-medium text-gray-600">Probability (Negative)</p>
                        <p className="text-2xl font-semibold text-gray-800"> {(parseFloat(response.probability["negative (class 1)"]) * 100).toFixed(2)}% </p>
                      </div>
                    </div>
                    <div className="mt-6 text-center">
                      <button type="button" onClick={handleGenerateReport} disabled={isReportLoading} className="bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-400 text-white font-bold py-3 px-6 rounded-lg text-base transition-all transform hover:scale-105 shadow-md flex items-center gap-2 justify-center mx-auto">
                        {isReportLoading ? <><Loader2 className="h-5 w-5 animate-spin" /> Processing Report...</> : <><FileText className="h-5 w-5" /> Generate Full Report</>}
                      </button>
                      {pdfUrl && (
                        <div className="mt-8 border-t pt-6">
                          <h4 className="text-xl font-bold text-gray-800 mb-4">Generated Report Preview</h4>
                          <div className="border rounded-lg overflow-hidden shadow-md"> <iframe src={pdfUrl} width="100%" height="600px" title="PDF Report Preview" /> </div>
                          <a href={pdfUrl} download="Gallstone_Analysis_Report.pdf" className="mt-4 bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-6 rounded-lg text-base transition-all inline-flex items-center gap-2 shadow-md"> <Download className="h-5 w-5" /> Download PDF </a>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
          </main>
        </div>
      </div>
    </div>
  );
}