"use client";

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { Button } from "./ui/button";
import { Upload, Users, AlertCircle, Loader2 } from "lucide-react";
import { Alert, AlertDescription } from "./ui/alert";
import { Progress } from "./ui/progress";

const BACKEND_URL = 'http://localhost:8000';

const ImageUpload = ({ onImageSelect, imageUrl, label, isLoading }) => {
  return (
    <div className="flex flex-col items-center space-y-4">
      <div 
        className={`border-2 border-dashed rounded-lg p-4 w-64 h-64 flex flex-col items-center justify-center cursor-pointer hover:bg-gray-50 transition ${isLoading ? 'opacity-50' : ''}`}
        onClick={() => !isLoading && document.getElementById(label).click()}
      >
        {imageUrl ? (
          <img 
            src={imageUrl} 
            alt="Preview" 
            className="max-w-full max-h-full object-contain"
          />
        ) : (
          <div className="text-center">
            {isLoading ? (
              <Loader2 className="mx-auto h-12 w-12 text-gray-400 animate-spin" />
            ) : (
              <>
                <Upload className="mx-auto h-12 w-12 text-gray-400" />
                <p className="mt-2 text-sm text-gray-600">Click to upload image</p>
              </>
            )}
          </div>
        )}
      </div>
      <input
        id={label}
        type="file"
        className="hidden"
        accept="image/*"
        onChange={onImageSelect}
        disabled={isLoading}
      />
      <p className="text-sm text-gray-600">{label}</p>
    </div>
  );
};

const PairComparison = () => {
  const [image1, setImage1] = useState(null);
  const [image2, setImage2] = useState(null);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageSelect = (setter) => async (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => setter(reader.result);
      reader.readAsDataURL(file);
    }
  };

  const analyzeKinship = async () => {
    try {
      setIsLoading(true);
      setError(null);
      setResult(null);

      // Convert base64 images to files
      const image1File = await fetch(image1)
        .then(res => res.blob())
        .then(blob => new File([blob], "image1.jpg", { type: "image/jpeg" }));
        
      const image2File = await fetch(image2)
        .then(res => res.blob())
        .then(blob => new File([blob], "image2.jpg", { type: "image/jpeg" }));
      
      const formData = new FormData();
      formData.append('image1', image1File);
      formData.append('image2', image2File);
      
      const response = await fetch(`${BACKEND_URL}/verify-kinship/`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to analyze kinship');
      }
      
      const data = await response.json();
      setResult({
        isKin: data.isKin,
        confidence: data.confidence,
        similarity: data.similarity,
        processingTime: data.processingTime
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
      
      <div className="flex flex-col md:flex-row justify-center items-center gap-8">
        <ImageUpload 
          onImageSelect={handleImageSelect(setImage1)}
          imageUrl={image1}
          label="First Person"
          isLoading={isLoading}
        />
        <ImageUpload 
          onImageSelect={handleImageSelect(setImage2)}
          imageUrl={image2}
          label="Second Person"
          isLoading={isLoading}
        />
      </div>
      
      <div className="flex justify-center">
        <Button 
          onClick={analyzeKinship}
          disabled={!image1 || !image2 || isLoading}
          className="w-48"
        >
          {isLoading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Analyzing...
            </>
          ) : (
            'Analyze Kinship'
          )}
        </Button>
      </div>

      {result && (
        <div className="space-y-4">
          <Alert className={result.isKin ? "bg-green-50" : "bg-blue-50"}>
            <AlertCircle className={`h-4 w-4 ${result.isKin ? "text-green-600" : "text-blue-600"}`} />
            <AlertDescription>
              {result.isKin 
                ? "These individuals appear to be related" 
                : "These individuals do not appear to be related"
              }
              <br />
              <span className="text-sm text-gray-600">
                Confidence: {(result.confidence * 100).toFixed(1)}%
                <br />
                Similarity Score: {(result.similarity).toFixed(3)}
                <br />
                Processing Time: {result.processingTime.toFixed(2)}s
              </span>
            </AlertDescription>
          </Alert>
          
          <div className="space-y-2">
            <div className="text-sm font-medium">Similarity Score</div>
            <Progress value={Math.max(0, Math.min(100, (result.similarity + 1) * 50))} />
          </div>
        </div>
      )}
    </div>
  );
};

const GroupAnalysis = () => {
  const [groupImage, setGroupImage] = useState(null);
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageSelect = async (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => setGroupImage(reader.result);
      reader.readAsDataURL(file);
    }
  };

  const analyzeGroup = async () => {
    try {
      setIsLoading(true);
      setError(null);
      setResults(null);

      // Convert base64 image to file
      const imageFile = await fetch(groupImage)
        .then(res => res.blob())
        .then(blob => new File([blob], "group.jpg", { type: "image/jpeg" }));
      
      const formData = new FormData();
      formData.append('image', imageFile);
      
      const response = await fetch(`${BACKEND_URL}/analyze-group/`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to analyze group photo');
      }
      
      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <div className="flex justify-center">
        <div 
          className={`border-2 border-dashed rounded-lg p-4 w-full max-w-2xl h-96 flex flex-col items-center justify-center cursor-pointer hover:bg-gray-50 transition ${isLoading ? 'opacity-50' : ''}`}
          onClick={() => !isLoading && document.getElementById("group-photo").click()}
        >
          {groupImage ? (
            <img 
              src={groupImage} 
              alt="Group Preview" 
              className="max-w-full max-h-full object-contain"
            />
          ) : (
            <div className="text-center">
              {isLoading ? (
                <Loader2 className="mx-auto h-12 w-12 text-gray-400 animate-spin" />
              ) : (
                <>
                  <Users className="mx-auto h-12 w-12 text-gray-400" />
                  <p className="mt-2 text-sm text-gray-600">Click to upload group photo</p>
                </>
              )}
            </div>
          )}
        </div>
        <input
          id="group-photo"
          type="file"
          className="hidden"
          accept="image/*"
          onChange={handleImageSelect}
          disabled={isLoading}
        />
      </div>

      <div className="flex justify-center">
        <Button 
          onClick={analyzeGroup}
          disabled={!groupImage || isLoading}
          className="w-48"
        >
          {isLoading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Analyzing...
            </>
          ) : (
            'Analyze Group'
          )}
        </Button>
      </div>

      {results && (
        <div className="space-y-4">
          <Alert variant="default">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              Detected {results.totalFaces} faces in the image
              <br />
              <span className="text-sm text-gray-600">
                Processing Time: {results.processingTime.toFixed(2)}s
              </span>
            </AlertDescription>
          </Alert>

          {results.results.map((result, idx) => (
            <Alert key={idx} className="bg-blue-50">
              <AlertCircle className="h-4 w-4 text-blue-600" />
              <AlertDescription>
                Person {result.pair[0]} and Person {result.pair[1]} appear to be related
                <br />
                <span className="text-sm text-gray-600">
                  Confidence: {(result.confidence * 100).toFixed(1)}%
                  <br />
                  Similarity Score: {result.similarity.toFixed(3)}
                </span>
              </AlertDescription>
            </Alert>
          ))}
        </div>
      )}
    </div>
  );
};

const KinshipAnalyzer = () => {
  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader>
        <CardTitle className="text-center">Kinship Verification System</CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="pair" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="pair">Pair Comparison</TabsTrigger>
            <TabsTrigger value="group">Group Analysis</TabsTrigger>
          </TabsList>
          <TabsContent value="pair">
            <PairComparison />
          </TabsContent>
          <TabsContent value="group">
            <GroupAnalysis />
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default KinshipAnalyzer;