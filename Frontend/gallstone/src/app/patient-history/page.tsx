"use client";

import React, { useState, useEffect } from "react";
import { useQuery } from "convex/react";
import { api } from "../../../convex/_generated/api";
import { type Id } from "../../../convex/_generated/dataModel";
import { 
  Loader2, 
  User, 
  FileText, 
  Phone, 
  MapPin, 
  AlertTriangle, 
  FileSearch, 
  CheckCircle, 
  XCircle,
  LogIn,
  X
} from "lucide-react";

// Type for report with URL - adjust this based on your actual API response structure
type ReportWithUrl = {
  _id: string;
  patientDetails: {
    name: string;
    address: string;
    phone: string;
  };
  predictionResult: {
    prediction_label: string;
    probability: {
      "positive (class 0)": string;
      "negative (class 1)": string;
    };
  };
  pdfUrl?: string;
  _creationTime: number;
};

export default function HistoryPage(): React.ReactElement {
  const [userId, setUserId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedReport, setSelectedReport] = useState<ReportWithUrl | null>(null);

  // On component mount, retrieve the userId from localStorage
  useEffect(() => {
    const checkUserId = () => {
      try {
        const storedUserId = localStorage.getItem("userId");
        setUserId(storedUserId);
      } catch (error) {
        console.error("Error accessing localStorage:", error);
        setUserId(null);
      } finally {
        setIsLoading(false);
      }
    };

    // Small delay to ensure localStorage is available
    setTimeout(checkUserId, 100);
  }, []);

  // Fetch reports using the Convex query
  // The query is "skipped" until the userId is loaded from localStorage
  const reports = useQuery(
    api.report.getReportsForUser,
    userId ? { userId: userId as Id<"users"> } : "skip"
  );

  // Handler to open the details modal
  const handleViewDetails = (report: ReportWithUrl) => {
    setSelectedReport(report);
    setIsModalOpen(true);
  };

  // Handler to close the modal
  const handleCloseModal = () => {
    setIsModalOpen(false);
    setSelectedReport(null);
  };

  // Handler to redirect to sign in
  const handleSignIn = () => {
    // Adjust this path based on your routing structure
    window.location.href = "/signin";
  };

  // Format date helper
  const formatDate = (timestamp: number) => {
    return new Date(timestamp).toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit"
    });
  };

  const renderContent = () => {
    // State 1: Initial loading (checking localStorage)
    if (isLoading) {
      return (
        <div className="flex flex-col items-center justify-center text-center text-gray-500 mt-20">
          <Loader2 className="h-12 w-12 animate-spin text-blue-600 mb-4" />
          <p className="text-xl font-semibold">Initializing...</p>
        </div>
      );
    }

    // State 2: No user ID found - show sign in prompt
    if (!userId) {
      return (
        <div className="flex flex-col items-center justify-center text-center text-gray-500 mt-20 mx-4">
          <div className="bg-blue-50 p-8 rounded-2xl max-w-md w-full">
            <LogIn className="h-16 w-16 text-blue-500 mb-6 mx-auto" />
            <h2 className="text-2xl font-bold text-blue-800 mb-4">Sign In Required</h2>
            <p className="text-gray-600 mb-6">
              Please sign in to view your prediction history and access your reports.
            </p>
            <button
              onClick={handleSignIn}
              className="w-full bg-blue-600 text-white font-semibold py-3 px-6 rounded-lg hover:bg-blue-700 transition-colors flex items-center justify-center gap-2"
            >
              <LogIn className="h-5 w-5" />
              Sign In
            </button>
          </div>
        </div>
      );
    }

    // State 3: Loading reports from the database
    if (reports === undefined) {
      return (
        <div className="flex flex-col items-center justify-center text-center text-gray-500 mt-20">
          <Loader2 className="h-12 w-12 animate-spin text-blue-600 mb-4" />
          <p className="text-xl font-semibold">Loading History...</p>
          <p className="text-sm text-gray-400 mt-2">Fetching your reports...</p>
        </div>
      );
    }

    // State 4: No reports found for this user
    if (reports.length === 0) {
      return (
        <div className="flex flex-col items-center justify-center text-center text-gray-500 mt-20 mx-4">
          <div className="bg-gray-50 p-8 rounded-2xl max-w-md w-full">
            <FileSearch className="h-16 w-16 text-gray-400 mb-6 mx-auto" />
            <h2 className="text-2xl font-bold text-gray-700 mb-4">No Reports Found</h2>
            <p className="text-gray-500">
              You have not generated any prediction reports yet. Create your first gallstone risk assessment to see it here.
            </p>
          </div>
        </div>
      );
    }

    // State 5: Display the list of reports
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
        {reports.map((report) => (
          <div key={report._id} className="bg-white rounded-xl shadow-md p-4 sm:p-6 flex flex-col justify-between hover:shadow-lg hover:ring-2 hover:ring-blue-500 transition-all">
            <div>
              <div className="flex items-center gap-3 mb-4">
                <div className={`p-2 rounded-full flex-shrink-0 ${
                  report.predictionResult.prediction_label === "Positive" ? "bg-red-100" : "bg-green-100"
                }`}>
                  {report.predictionResult.prediction_label === "Positive" ? 
                    <XCircle className="h-5 w-5 sm:h-6 sm:w-6 text-red-600" /> : 
                    <CheckCircle className="h-5 w-5 sm:h-6 sm:w-6 text-green-600" />}
                </div>
                <div className="min-w-0 flex-1">
                  <h3 className="text-lg sm:text-xl font-bold text-gray-800 truncate">{report.patientDetails.name}</h3>
                  {report._creationTime && (
                    <p className="text-xs text-gray-500">{formatDate(report._creationTime)}</p>
                  )}
                </div>
              </div>
              
              <div className="space-y-3 text-sm text-gray-600">
                <div className="mb-3">
                  <span className={`inline-block font-semibold py-1 px-3 rounded-full text-xs sm:text-sm ${
                    report.predictionResult.prediction_label === 'Positive' 
                      ? 'bg-red-100 text-red-800' 
                      : 'bg-green-100 text-green-800'
                  }`}>
                    {report.predictionResult.prediction_label} for Gallstones
                  </span>
                </div>
                
                <p className="flex items-start gap-2">
                  <MapPin className="h-4 w-4 mt-0.5 flex-shrink-0 text-gray-400" /> 
                  <span className="break-words">{report.patientDetails.address}</span>
                </p>
                
                <p className="flex items-center gap-2">
                  <Phone className="h-4 w-4 flex-shrink-0 text-gray-400" /> 
                  <span>{report.patientDetails.phone}</span>
                </p>
              </div>
            </div>
            
            <button
              onClick={() => handleViewDetails(report)}
              className="mt-6 w-full bg-blue-600 text-white font-semibold py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors flex items-center justify-center gap-2 text-sm sm:text-base"
            >
              <FileText className="h-4 w-4 sm:h-5 sm:w-5" />
              View Details & PDF
            </button>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-6 sm:py-8 max-w-7xl">
        <div className="text-center mb-8 sm:mb-10">
          <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-2">Prediction History</h1>
          <p className="text-base sm:text-lg text-gray-600">Review your past gallstone risk assessment reports.</p>
        </div>
        {renderContent()}
      </div>

      {/* Details Modal */}
      {isModalOpen && selectedReport && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-60 p-4">
          <div className="bg-white rounded-2xl shadow-2xl w-full max-w-4xl max-h-[90vh] flex flex-col">
            {/* Modal Header */}
            <div className="p-4 sm:p-6 border-b flex justify-between items-center">
              <h2 className="text-xl sm:text-2xl font-bold text-gray-900">Report Details</h2>
              <button 
                onClick={handleCloseModal} 
                className="text-gray-500 hover:text-gray-800 p-1 hover:bg-gray-100 rounded-full transition-colors"
                aria-label="Close modal"
              >
                <X className="h-6 w-6" />
              </button>
            </div>
            
            {/* Modal Content */}
            <div className="p-4 sm:p-6 overflow-y-auto">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Left side: Details */}
                <div className="space-y-6">
                  <div>
                    <h4 className="font-bold text-lg mb-3 text-gray-800 border-b pb-2">Patient Information</h4>
                    <div className="space-y-2 text-sm sm:text-base">
                      <p><strong className="text-gray-700">Name:</strong> <span className="text-gray-600">{selectedReport.patientDetails.name}</span></p>
                      <p><strong className="text-gray-700">Address:</strong> <span className="text-gray-600">{selectedReport.patientDetails.address}</span></p>
                      <p><strong className="text-gray-700">Phone:</strong> <span className="text-gray-600">{selectedReport.patientDetails.phone}</span></p>
                      {selectedReport._creationTime && (
                        <p><strong className="text-gray-700">Generated:</strong> <span className="text-gray-600">{formatDate(selectedReport._creationTime)}</span></p>
                      )}
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-bold text-lg mb-3 text-gray-800 border-b pb-2">Prediction Outcome</h4>
                    <div className="space-y-2 text-sm sm:text-base">
                      <p><strong className="text-gray-700">Result:</strong> 
                        <span className={`ml-2 font-bold ${
                          selectedReport.predictionResult.prediction_label === 'Positive' ? 'text-red-600' : 'text-green-600'
                        }`}>
                          {selectedReport.predictionResult.prediction_label}
                        </span>
                      </p>
                      <p><strong className="text-gray-700">Positive Probability:</strong> 
                        <span className="text-gray-600 ml-2">
                          {(parseFloat(selectedReport.predictionResult.probability["positive (class 0)"]) * 100).toFixed(2)}%
                        </span>
                      </p>
                      <p><strong className="text-gray-700">Negative Probability:</strong> 
                        <span className="text-gray-600 ml-2">
                          {(parseFloat(selectedReport.predictionResult.probability["negative (class 1)"]) * 100).toFixed(2)}%
                        </span>
                      </p>
                    </div>
                  </div>
                </div>
                
                {/* Right side: PDF Preview */}
                <div>
                  <h4 className="font-bold text-lg mb-3 text-gray-800 border-b pb-2">Generated Report PDF</h4>
                  {selectedReport.pdfUrl ? (
                    <div className="border rounded-lg overflow-hidden">
                      <iframe 
                        src={selectedReport.pdfUrl} 
                        width="100%" 
                        height="500px" 
                        title="PDF Report Preview" 
                        className="w-full"
                      />
                    </div>
                  ) : (
                    <div className="border rounded-lg p-8 text-center">
                      <AlertTriangle className="h-12 w-12 text-amber-500 mx-auto mb-4" />
                      <p className="text-amber-600 font-semibold">PDF Preview Unavailable</p>
                      <p className="text-gray-500 text-sm mt-2">Could not load PDF preview</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
            
            {/* Modal Footer */}
            <div className="p-4 sm:p-6 bg-gray-50 border-t rounded-b-2xl">
              <div className="flex justify-end gap-3">
                {selectedReport.pdfUrl && (
                  <a
                    href={selectedReport.pdfUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="bg-blue-600 text-white font-semibold py-2 px-4 sm:px-6 rounded-lg hover:bg-blue-700 transition-colors text-sm sm:text-base"
                  >
                    Download PDF
                  </a>
                )}
                <button 
                  onClick={handleCloseModal} 
                  className="bg-gray-200 text-gray-800 font-semibold py-2 px-4 sm:px-6 rounded-lg hover:bg-gray-300 transition-colors text-sm sm:text-base"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}