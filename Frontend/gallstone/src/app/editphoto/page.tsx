"use client";

import React, { useState } from "react";
import { useMutation } from "convex/react";
import { api } from "../../../convex/_generated/api";
import { toast } from "sonner";
import { useRouter } from "next/navigation";

export default function EditPhoto() {
    const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const updatePhoto = useMutation(api.task.updateUserPhoto);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
    }
  };

  const handleUpload = async () => {
    if (!file) {
      toast.error("Please select an image first.");
      return;
    }

    setLoading(true);
    try {
      // Upload to Cloudinary
      const formData = new FormData();
      formData.append("file", file);
      formData.append("upload_preset", process.env.NEXT_PUBLIC_CLOUDINARY_PRESET!);

      const res = await fetch(
        `https://api.cloudinary.com/v1_1/${process.env.NEXT_PUBLIC_CLOUDINARY_CLOUD_NAME}/image/upload`,
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await res.json();
      if (!data.secure_url) throw new Error("Cloudinary upload failed");

      // Update DB with new photo link
      const userId = localStorage.getItem("userId");
      if (!userId) throw new Error("Not logged in");

      await updatePhoto({
        userId: userId as any,
        photo_link: data.secure_url,
      });

      toast.success("Profile photo updated!");
      router.push("/profile");
    } catch (err: any) {
      console.error(err);
      toast.error(err.message || "Upload failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-4">
      <h1 className="text-2xl font-bold mb-6">Edit Profile Photo</h1>

      {/* Preview */}
      {preview ? (
        <img
          src={preview}
          alt="Preview"
          className="w-40 h-40 rounded-full object-cover border mb-4"
        />
      ) : (
        <div className="w-40 h-40 rounded-full border bg-gray-100 flex items-center justify-center mb-4 text-gray-400">
          No Photo
        </div>
      )}

      {/* Custom File Input */}
      <label
        htmlFor="file-upload"
        className="cursor-pointer inline-block px-6 py-2 rounded-lg border border-gray-300 bg-white text-gray-700 hover:bg-gray-100 transition mb-4"
      >
        {file ? "Change File" : "Choose File"}
      </label>
      <input
        id="file-upload"
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        className="hidden"
      />

      {/* Upload Button */}
      <button
        onClick={handleUpload}
        disabled={loading}
        className="px-6 py-2 rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50"
      >
        {loading ? "Uploading..." : "Upload Photo"}
      </button>
    </div>
  );
}
