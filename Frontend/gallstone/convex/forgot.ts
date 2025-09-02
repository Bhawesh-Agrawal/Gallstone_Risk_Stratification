// convex/actions/userActions.ts
import { query, action, mutation, internalMutation } from "./_generated/server";
import { internal } from "./_generated/api";
import { v } from "convex/values";
import bcrypt from "bcryptjs";
import {api} from "../convex/_generated/api";

// Query: get user by email
export const getUserByEmail = query({
  args: { email: v.string() },
  handler: async (ctx, { email }) => {
    try {
      const user = await ctx.db
        .query("users")
        .filter((q) => q.eq(q.field("email"), email))
        .unique();
      return user || null;
    } catch (err) {
      console.error("Error fetching user by email:", err);
      return null;
    }
  },
});

// Action: update password (uses bcrypt which requires setTimeout)
export const updatePassword = action({
  args: { email: v.string(), newPassword: v.string() },
  handler: async (ctx, { email, newPassword }) => {
    try {
      // Hash password first (this uses setTimeout internally)
      const hashedPassword = await bcrypt.hash(newPassword, 10);

      // Fetch the user using runQuery
      const user = await ctx.runQuery(api.forgot.getUserByEmail, { email });

      if (!user) {
        return { success: false, message: "User not found" };
      }

      // Update password using runMutation
      await ctx.runMutation(internal.forgot.updateUserPassword, {
        userId: user._id,
        hashedPassword,
      });

      return { success: true, message: "Password updated successfully" };
    } catch (error) {
      console.error(error);
      return { success: false, message: (error as Error).message };
    }
  },
});

// Internal mutation to update user password
export const updateUserPassword = internalMutation({
  args: { 
    userId: v.id("users"), 
    hashedPassword: v.string() 
  },
  handler: async (ctx, { userId, hashedPassword }) => {
    await ctx.db.patch(userId, { password: hashedPassword });
  },
});