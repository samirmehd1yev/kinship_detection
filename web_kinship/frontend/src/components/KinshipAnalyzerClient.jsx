"use client";

import React, { useState } from 'react';
import Image from 'next/image';
import { Upload, Users } from "lucide-react";

const BACKEND_URL = 'http://localhost:8000';

export default function KinshipAnalyzerClient() {
  const [image1, setImage1] = useState(null);
  const [image2, setImage2] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleImageUpload = (setter) => (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => setter(reader.result);
      reader.readAsDataURL(file);
    }
  };

  const analyzeKinship = async () => {
    try {
      setLoading(true);
      setError(null);

      const formData = new FormData();
      
      // Convert base64 to files
      const image1File = await fetch(image1)
        .then(res => res.blob())
        .then(blob => new File([blob], "image1.jpg", { type: "image/jpeg" }));
      
      const image2File = await fetch(image2)
        .then(res => res.blob())
        .then(blob => new File([blob], "image2.jpg", { type: "image/jpeg" }));

      formData.append('image1', image1File);
      formData.append('image2', image2File);

      const response = await fetch(`${BACKEND_URL}/verify-kinship/`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to analyze images');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Image Upload Sections */}
        <div className="space-y-4">
          <div
            onClick={() => document.getElementById('image1').click()}
            className="border-2 border-dashed rounded-lg p-4 h-64 flex items-center justify-center cursor-pointer hover:bg-gray-50"
          >
            {image1 ? (
              <img src={image1} alt="First person" className="max-h-full w-auto" />
            ) : (
              <div className="text-center">
                <Upload className="mx-auto h-12 w-12 text-gray-400" />
                <p className="mt-2 text-sm text-gray-500">Upload first image</p>
              </div>
            )}
          </div>
          <input
            id="image1"
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleImageUpload(setImage1)}
          />
        </div>

        <div className="space-y-4">
          <div
            onClick={() => document.getElementById('image2').click()}
            className="border-2 border-dashed rounded-lg p-4 h-64 flex items-center justify-center cursor-pointer hover:bg-gray-50"
          >
            {image2 ? (
              <img src={image2} alt="Second person" className="max-h-full w-auto" />
            ) : (
              <div className="text-center">
                <Upload className="mx-auto h-12 w-12 text-gray-400" />
                <p className="mt-2 text-sm text-gray-500">Upload second image</p>
              </div>
            )}
          </div>
          <input
            id="image2"
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleImageUpload(setImage2)}
          />
        </div>
      </div>

      {/* Action Button */}
      <div className="flex justify-center">
        <button
          onClick={analyzeKinship}
          disabled={!image1 || !image2 || loading}
          className={`px-4 py-2 bg-blue-500 text-white rounded-lg ${
            (!image1 || !image2 || loading) ? 'opacity-50 cursor-not-allowed' : 'hover:bg-blue-600'
          }`}
        >
          {loading ? 'Analyzing...' : 'Analyze Kinship'}
        </button>
      </div>

      {/* Results */}
      {error && (
        <div className="p-4 bg-red-50 text-red-700 rounded-lg">
          {error}
        </div>
      )}

      {result && (
        <div className={`p-4 ${result.isKin ? 'bg-green-50' : 'bg-blue-50'} rounded-lg`}>
          <h3 className="text-lg font-semibold">
            {result.isKin ? 'Related' : 'Not Related'}
          </h3>
          <p className="text-sm mt-2">
            Confidence: {(result.confidence * 100).toFixed(1)}%
            <br />
            Similarity Score: {result.similarity.toFixed(3)}
            <br />
            Processing Time: {result.processingTime.toFixed(2)}s
          </p>
        </div>
      )}
    </div>
  );
}