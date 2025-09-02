"use client";

import React, { useState } from "react";
import Link from "next/link";
import { useAction } from "convex/react";
import { api } from "../../../convex/_generated/api";
import { toast } from "sonner";
import { useRouter } from "next/navigation";

export default function SignIn() {
  const router = useRouter();
  const signIn = useAction(api.user.signIn); // useAction for async actions

  const [formData, setFormData] = useState({ email: "", password: "" });
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const res = await signIn({
        email: formData.email,
        password: formData.password,
      });

      if (res.status === "success" && res.user) {
        toast.success(res.message || "Signed in successfully");
        localStorage.setItem("userId", res.user.id);
        router.push("/");
      } else {
        toast.error(res.message || "Invalid credentials");
      }
    } catch (err) {
      console.error(err);
      toast.error("Something went wrong. Try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="w-full max-w-md bg-white rounded-2xl shadow-lg p-8">
        {/* Logo */}
        <div className="flex justify-center mb-6">
          <div className="flex items-center gap-2">
            <span className="text-blue-600 font-extrabold text-2xl">
              Gallitify
            </span>
          </div>
        </div>

        {/* Heading */}
        <h2 className="text-center text-2xl font-bold text-gray-900 mb-1">
          Sign in
        </h2>
        <p className="text-center text-sm text-gray-500 mb-6">
          Use your Gallitify account
        </p>

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          <input
            type="email"
            placeholder="Email address"
            value={formData.email}
            onChange={(e) =>
              setFormData({ ...formData, email: e.target.value })
            }
            className="w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none text-gray-700"
            required
          />
          <input
            type="password"
            placeholder="Password"
            value={formData.password}
            onChange={(e) =>
              setFormData({ ...formData, password: e.target.value })
            }
            className="w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none text-gray-700"
            required
          />

          {/* Actions */}
          <div className="flex justify-between items-center text-sm text-blue-600">
            <Link href="/forgotpassword" className="hover:underline">
              Forgot password?
            </Link>
            <Link href="/createaccount" className="hover:underline">
              Create account
            </Link>
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={loading}
            className="w-full bg-blue-600 text-white font-semibold py-3 rounded-lg hover:bg-blue-700 transition disabled:opacity-50"
          >
            {loading ? "Signing in..." : "Sign In"}
          </button>
        </form>

        {/* Divider */}
        <div className="flex items-center my-6">
          <div className="flex-grow h-px bg-gray-200"></div>
          <span className="px-3 text-gray-400 text-sm">or</span>
          <div className="flex-grow h-px bg-gray-200"></div>
        </div>

        {/* Google-style social login */}
        <button className="w-full border flex items-center justify-center gap-3 py-3 rounded-lg hover:bg-gray-50 transition">
          <svg
            className="w-5 h-5"
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 48 48"
          >
            <path
              fill="#FFC107"
              d="M43.6 20.5H42V20H24v8h11.3c-1.6 4.6-6 8-11.3 8-6.6 0-12-5.4-12-12s5.4-12 12-12c3.1 0 6 .9 8.3 3l5.7-5.7C34.6 7.6 29.6 6 24 6c-9.9 0-18 8.1-18 18s8.1 18 18 18c9 0 17-6.5 17-18 0-1.2-.1-2.3-.4-3.5z"
            />
            <path
              fill="#FF3D00"
              d="M6.3 14.7l6.6 4.8C14.3 16.1 18.8 13 24 13c3.1 0 6 .9 8.3 3l5.7-5.7C34.6 7.6 29.6 6 24 6c-7.7 0-14.3 4.3-17.7 10.7z"
            />
            <path
              fill="#4CAF50"
              d="M24 44c5.2 0 10.1-1.8 13.8-5l-6.4-5.4C29.2 35.1 26.7 36 24 36c-5.3 0-9.7-3.4-11.3-8l-6.6 5.1C9.7 39.6 16.3 44 24 44z"
            />
            <path
              fill="#1976D2"
              d="M43.6 20.5H42V20H24v8h11.3c-1.1 3.2-3.6 5.9-6.7 7.2l6.4 5.4C38.1 38.9 41 32.1 41 24c0-1.2-.1-2.3-.4-3.5z"
            />
          </svg>
          <span className="text-gray-700 font-medium">Sign in with Google</span>
        </button>
      </div>
    </div>
  );
}
