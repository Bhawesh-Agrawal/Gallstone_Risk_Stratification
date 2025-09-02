// convex/auth.ts
import { action, ActionCtx } from "./_generated/server";
import { v } from "convex/values";
import bcrypt from "bcryptjs";
import { api } from "./_generated/api";

// Define argument and return types for clarity
type SignInArgs = {
  email: string;
  password: string;
};

type SignInResult =
  | { status: "success"; message: string; user: { id: string; name: string; email: string; profession: string; role: string; photo_link: string | null } }
  | { status: "error"; message: string };




export const signIn = action({
  args: {
    email: v.string(),
    password: v.string(),
  },
  handler: async (ctx: ActionCtx, { email, password }: SignInArgs): Promise<SignInResult> => {
    // 1. Get user by email
    const user = await ctx.runQuery(api.task.getUserByEmail, { email });
    if (!user) {
      return { status: "error", message: "User not found" };
    }

    // 2. Compare password
    const isPasswordValid = await bcrypt.compare(password, user.password);
    if (!isPasswordValid) {
      return { status: "error", message: "Invalid credentials" };
    }

    // 3. Update last login
    await ctx.runMutation(api.task.updateLastLogin, {
      userId: user._id,
      lastLogin: new Date().toISOString(),
    });

    return {
      status: "success",
      message: "Login successful",
      user: {
        id: user._id,
        name: user.name,
        email: user.email,
        profession: user.profession,
        role: user.role,
        photo_link: user.photo_link ?? null,
      },
    };
  },
});
