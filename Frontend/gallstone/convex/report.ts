// convex/report.ts

import { action, internalMutation, query } from "./_generated/server";
import { internal } from "./_generated/api";
import { v } from "convex/values";
import { Id } from "./_generated/dataModel";

export const saveReport = action({
  args: {
    userId: v.id("users"), 
    patientDetails: v.object({
      name: v.string(),
      address: v.string(),
      phone: v.string(),
    }),
    formData: v.any(),
    predictionResult: v.any(),
    pdfBase64: v.string(),
  },
  handler: async (ctx, args) => {
    // ✅ DECODE BASE64 USING WEB APIs INSTEAD OF NODE.JS BUFFER
    const binaryString = atob(args.pdfBase64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    const pdfBlob = new Blob([bytes], { type: "application/pdf" });

    // 2. Store the PDF blob in Convex File Storage
    const storageId = await ctx.storage.store(pdfBlob);

    // 3. Call an internal mutation to write the final record to the database
    await ctx.runMutation(internal.report.internalSaveReport, {
      userId: args.userId,
      patientDetails: args.patientDetails,
      formData: args.formData,
      predictionResult: args.predictionResult,
      pdfStorageId: storageId,
    });
  },
});

export const internalSaveReport = internalMutation({
  args: {
    userId: v.id("users"),
    patientDetails: v.object({
      name: v.string(),
      address: v.string(),
      phone: v.string(),
    }),
    formData: v.any(),
    predictionResult: v.any(),
    pdfStorageId: v.id("_storage"),
  },
  handler: async (ctx, args) => {
    await ctx.db.insert("reports", {
      userId: args.userId,
      patientDetails: args.patientDetails,
      formData: args.formData,
      predictionResult: args.predictionResult,
      pdfStorageId: args.pdfStorageId,
    });
  },
});

export const getReportsForUser = query({
  // Define the argument: the ID of the user whose reports we want to fetch.
  args: {
    userId: v.id("users"),
  },
  handler: async (ctx, args) => {
    // 1. Fetch all reports from the database that match the provided userId.
    // We use the 'by_userId' index defined in the schema for maximum efficiency.
    const reports = await ctx.db
      .query("reports")
      .withIndex("by_userId", (q) => q.eq("userId", args.userId))
      .collect();

    // 2. For each report, generate a temporary, accessible URL for its PDF
    // stored in Convex file storage.
    const reportsWithPdfUrls = await Promise.all(
      reports.map(async (report) => {
        // The getUrl method returns null if the storage ID is invalid.
        const pdfUrl = await ctx.storage.getUrl(report.pdfStorageId);
        return {
          ...report,
          pdfUrl: pdfUrl || "", // Return the URL or an empty string.
        };
      })
    );

    // 3. Return the enhanced list of reports to the client.
    return reportsWithPdfUrls;
  },
});