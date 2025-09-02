"use client";

import React, { useEffect, useState, useMemo } from "react";
import { useQuery } from "convex/react";
import { api } from "../../../convex/_generated/api";
import { Id } from "../../../convex/_generated/dataModel";
import Link from "next/link";

export default function Profile() {
  const [userId, setUserId] = useState<Id<"users"> | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);

  useEffect(() => {
    const storedUserId = localStorage.getItem("userId");
    setUserId(storedUserId as Id<"users"> | null);
    setIsInitialized(true);
  }, []);

  const userData = useQuery(
    api.task.getUserById,
    userId ? { userId } : "skip"
  );

  const isLoading = useMemo(
    () => !isInitialized || (userId && userData === undefined),
    [isInitialized, userId, userData]
  );

  if (isInitialized && !userId) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center">
        <h2 className="text-xl font-bold mb-4">Not Logged In</h2>
        <a href="/signin" className="text-blue-600 mb-2">Sign In</a>
        <a href="/createaccount" className="text-gray-600">Create Account</a>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-gray-600">Loading profile...</p>
      </div>
    );
  }

  if (!userData) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center">
        <h2 className="text-xl font-bold mb-4">Profile Not Found</h2>
        <a href="/signin" className="text-blue-600">Sign In Again</a>
      </div>
    );
  }

  const memberSince = new Date(userData.created_date).toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });

  const profilePhoto =
    userData.photo_link && userData.photo_link.trim() !== ""
      ? userData.photo_link
      : "https://plus.unsplash.com/premium_vector-1727958429097-aed514edc834?q=80&w=880&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D";

  return (
    <div className="min-h-screen flex flex-col items-center py-12 px-6">
      {/* Profile Photo + Edit Button */}
      <div className="flex flex-col items-center mb-8">
        <img
          src={profilePhoto}
          alt={`${userData.name}'s profile`}
          className="w-40 h-40 rounded-full border bg-gray-100 object-cover"
          onError={(e) => {
            const target = e.target as HTMLImageElement;
            target.src = "https://plus.unsplash.com/premium_vector-1727958429097-aed514edc834?q=80&w=880&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D";
          }}
        />
        <Link href="/editphoto">
          <button className="mt-3 px-4 py-1 text-sm border rounded-full text-gray-600 hover:bg-gray-100">
          Edit Photo
            </button>
        </Link>
        
      </div>

      {/* Info Section */}
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-bold text-gray-900 capitalize">{userData.name}</h1>
        <p className="text-gray-700">{userData.profession}</p>
        <p className="text-gray-500 text-sm">Member since {memberSince}</p>
        <p className="text-gray-800 mt-2">{userData.email}</p>
      </div>

      {/* Action Buttons */}
      <div className="flex space-x-6 mt-10">
        <a href="/forgotpassword" className="text-blue-600 hover:underline">
          Forgot Password
        </a>
        <a href="/signout" className="text-red-600 hover:underline">
          Sign Out
        </a>
      </div>
    </div>
  );
}
