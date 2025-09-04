// Add these functions to your Convex chat.ts file

import { v } from "convex/values";
import { mutation, query } from "./_generated/server";

// Get all conversations for a user
export const getUserConversations = query({
  args: { userId: v.string() },
  handler: async (ctx, args) => {
    const conversations = await ctx.db
      .query("conversations")
      .withIndex("by_userId", (q) => q.eq("userId", args.userId))
      .order("desc") // Most recent first
      .collect();

    return conversations;
  },
});

// Get a specific conversation by ID
export const getConversationById = query({
  args: { conversationId: v.id("conversations") },
  handler: async (ctx, args) => {
    const conversation = await ctx.db.get(args.conversationId);
    return conversation;
  },
});

// Create a new conversation
export const createConversation = mutation({
  args: {
    userId: v.string(),
    title: v.string(),
    history: v.array(v.object({
      role: v.union(v.literal("user"), v.literal("assistant")),
      content: v.string(),
    })),
  },
  handler: async (ctx, args) => {
    const conversationId = await ctx.db.insert("conversations", {
      userId: args.userId,
      title: args.title,
      history: args.history,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    });
    return conversationId;
  },
});

// Update conversation history (existing function, but updated for new schema)
export const updateConversation = mutation({
  args: {
    conversationId: v.id("conversations"),
    history: v.array(v.object({
      role: v.union(v.literal("user"), v.literal("assistant")),
      content: v.string(),
    })),
  },
  handler: async (ctx, args) => {
    await ctx.db.patch(args.conversationId, {
      history: args.history,
      updatedAt: Date.now(),
    });
  },
});

// Update conversation title
export const updateConversationTitle = mutation({
  args: {
    conversationId: v.id("conversations"),
    title: v.string(),
  },
  handler: async (ctx, args) => {
    await ctx.db.patch(args.conversationId, {
      title: args.title,
      updatedAt: Date.now(),
    });
  },
});

// Delete a conversation
export const deleteConversation = mutation({
  args: { conversationId: v.id("conversations") },
  handler: async (ctx, args) => {
    await ctx.db.delete(args.conversationId);
  },
});

// Your existing function (keep this)
export const getConversation = query({
  args: { userId: v.string() },
  handler: async (ctx, args) => {
    const conversation = await ctx.db
      .query("conversations")
      .withIndex("by_userId", (q) => q.eq("userId", args.userId))
      .unique();

    return conversation ? conversation.history : [];
  },
});