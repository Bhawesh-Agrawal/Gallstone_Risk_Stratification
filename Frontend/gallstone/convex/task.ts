import { query, mutation, action } from "./_generated/server";
import { v } from "convex/values";
import { api } from "./_generated/api";
import type { Id } from "./_generated/dataModel";
import bcrypt from "bcryptjs";


export const updateLastLogin = mutation({
  args: { userId: v.id("users"), lastLogin: v.string() },
  handler: async (ctx, { userId, lastLogin }) => {
    await ctx.db.patch(userId, { last_login: lastLogin });
  },
});

export const updateUserPhoto = mutation({
  args: {
    userId: v.id("users"),
    photo_link: v.string(),
  },
  handler: async (ctx, { userId, photo_link }) => {
    await ctx.db.patch(userId, { photo_link });
    return { success: true };
  },
});

export const getUserById = query({
  args: { userId: v.id("users") },
  handler: async (ctx, { userId }) => {
    try {
      const user = await ctx.db.get(userId);
      return user || null;
    } catch (error) {
      console.error("Error fetching user by ID:", error);
      return null;
    }
  },
});


export const getUserByEmail = query({
  args: { email: v.string() },
  handler: async (ctx, { email }) => {
    try {
      const user = await ctx.db
        .query("users")
        .filter((q) => q.eq(q.field("email"), email))
        .unique();
      return user || null;
    } catch (error) {
      console.error("Error fetching user by email:", error);
      return null; // safer fallback
    }
  },
});

// ✅ Mutation: insert new user
export const insertUser = mutation({
  args: {
    name: v.string(),
    profession: v.string(),
    email: v.string(),
    password: v.string(),
    photo_link: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    try {
      return await ctx.db.insert("users", {
        ...args,
        created_date: new Date().toISOString(),
        isVerified: false,
        role: "user",
      });
    } catch (error) {
      console.error("Error inserting user:", error);
      throw new Error("Failed to create user in database");
    }
  },
});

// ✅ Action: safe user creation with consistent response
export const createUser = action({
  args: {
    name: v.string(),
    profession: v.string(),
    email: v.string(),
    password: v.string(),
    photo_link: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    try {
      // Check if user already exists
      const existingUser = await ctx.runQuery(api.task.getUserByEmail, {
        email: args.email,
      });

      if (existingUser) {
        return {
          success: false,
          message: "User already exists with this email",
          code: "USER_EXISTS",
        };
      }

      // Hash password
      let hashedPassword: string;
      try {
        hashedPassword = await bcrypt.hash(args.password, 10);
      } catch (err) {
        console.error("Error hashing password:", err);
        return {
          success: false,
          message: "Failed to process password",
          code: "HASH_ERROR",
        };
      }

      // Insert user
      const newUserId: Id<"users"> = await ctx.runMutation(api.task.insertUser, {
        name: args.name,
        profession: args.profession,
        email: args.email,
        password: hashedPassword,
        photo_link: args.photo_link,
      });

      return {
        success: true,
        message: "Account created successfully",
        code: "USER_CREATED",
        data: { userId: newUserId },
      };
    } catch (error) {
      console.error("Error creating user:", error);
      return {
        success: false,
        message: "Unexpected error occurred while creating user",
        code: "SERVER_ERROR",
      };
    }
  },
});


// ---- Type helpers ----
export type GetUserByEmail = typeof getUserByEmail;
export type InsertUser = typeof insertUser;
export type CreateUser = typeof createUser;
