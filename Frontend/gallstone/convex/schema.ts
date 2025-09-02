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
});
