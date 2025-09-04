import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";

export default defineSchema({
  users: defineTable({
    name: v.string(),
    profession: v.string(),
    email: v.string(),
    password: v.string(),
    photo_link: v.optional(v.string()),
    created_date: v.string(),
    last_login: v.optional(v.string()), 
    isVerified: v.boolean(),
    role: v.string(),
  })
    .index("by_email", ["email"]),

  reports: defineTable({
    userId: v.id("users"),
    patientDetails: v.object({
      name: v.string(),
      address: v.string(),
      phone: v.string(),
    }),
    formData: v.any(),
    predictionResult: v.any(),
    pdfStorageId: v.id("_storage"),
  })
    .index("by_userId", ["userId"]),

  conversations: defineTable({
    userId: v.string(),
    title: v.string(),
    history: v.array(v.object({
      role: v.union(v.literal("user"), v.literal("assistant")),
      content: v.string(),
    })),
    createdAt: v.number(),
    updatedAt: v.number(),
  }).index("by_userId", ["userId"]),
});