"use client";

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Upload, Users, AlertCircle, Loader2 } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";

const BACKEND_URL = 'http://localhost:8000';

const ImageUpload = ({ onImageSelect, imageUrl, label, isLoading }) => {
  return (
    <div className="flex flex-col items-center space-y-4">
      <div 
        className={`border-2 border-dashed rounded-lg p-4 w-64 h-64 flex flex-col items-center justify-center cursor-pointer hover:bg-gray-50 transition-colors ${isLoading ? 'opacity-50' : ''}`}
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
      <p className="text-sm font-medium text-gray-700">{label}</p>
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

      const formData = new FormData();
      
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
        throw new Error(errorData.detail || 'Failed to analyze kinship');
      }
      
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {error && (
        <Alert variant="destructive" className="bg-red-50 border-red-200">
          <AlertCircle className="h-4 w-4 text-red-600" />
          <AlertDescription className="text-red-700">{error}</AlertDescription>
        </Alert>
      )}
      
      <div className="flex flex-col md:flex-row justify-center items-center gap-8">
        <div className="space-y-2">
          <ImageUpload 
            onImageSelect={handleImageSelect(setImage1)}
            imageUrl={image1}
            label="First Person"
            isLoading={isLoading}
          />
          {result?.gender1 && (
            <p className="text-center text-sm font-medium text-gray-600">
              {result.gender1}
            </p>
          )}
        </div>
        <div className="space-y-2">
          <ImageUpload 
            onImageSelect={handleImageSelect(setImage2)}
            imageUrl={image2}
            label="Second Person"
            isLoading={isLoading}
          />
          {result?.gender2 && (
            <p className="text-center text-sm font-medium text-gray-600">
              {result.gender2}
            </p>
          )}
        </div>
      </div>
      
      <div className="flex justify-center">
        <Button 
          onClick={analyzeKinship}
          disabled={!image1 || !image2 || isLoading}
          className="w-48 bg-blue-600 hover:bg-blue-700 text-white"
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
          <Alert className={result.isKin ? "bg-green-50 border-green-200" : "bg-blue-50 border-blue-200"}>
            <AlertCircle className={`h-4 w-4 ${result.isKin ? "text-green-600" : "text-blue-600"}`} />
            <AlertDescription>
              <p className={`font-semibold ${result.isKin ? "text-green-700" : "text-blue-700"}`}>
                {result.isKin 
                  ? "These individuals appear to be related" 
                  : "These individuals do not appear to be related"
                }
              </p>
              <div className="mt-2 text-sm text-gray-600 space-y-1">
                <p>Confidence: {(result.confidence * 100).toFixed(1)}%</p>
                <p>Similarity Score: {(result.similarity).toFixed(3)}</p>
                <p>Processing Time: {result.processingTime.toFixed(2)}s</p>
              </div>
            </AlertDescription>
          </Alert>

          {result.isKin && result.relationships && (
            <div className="space-y-3">
              <h3 className="text-lg font-semibold text-gray-700">Predicted Relationships:</h3>
              {result.relationships.map((rel, idx) => (
                <Alert 
                  key={idx} 
                  className={`${idx === 0 ? 'bg-green-50 border-green-200' : 'bg-yellow-50 border-yellow-200'}`}
                >
                  <AlertCircle className={`h-4 w-4 ${idx === 0 ? 'text-green-600' : 'text-yellow-600'}`} />
                  <AlertDescription>
                    <p className={`font-medium ${idx === 0 ? 'text-green-700' : 'text-yellow-700'}`}>
                      {rel.description}
                    </p>
                    <p className="text-sm text-gray-600">
                      Confidence: {(rel.confidence * 100).toFixed(1)}%
                    </p>
                  </AlertDescription>
                </Alert>
              ))}
            </div>
          )}
          
          <div className="space-y-2">
            <div className="text-sm font-medium text-gray-700">Similarity Score</div>
            <Progress 
              value={Math.max(0, Math.min(100, (result.similarity + 1) * 50))}
              className="h-2 bg-gray-100"
            />
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

      const formData = new FormData();
      const imageFile = await fetch(groupImage)
        .then(res => res.blob())
        .then(blob => new File([blob], "group.jpg", { type: "image/jpeg" }));
      
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
        <Alert variant="destructive" className="bg-red-50 border-red-200">
          <AlertCircle className="h-4 w-4 text-red-600" />
          <AlertDescription className="text-red-700">{error}</AlertDescription>
        </Alert>
      )}

      <div className="space-y-4">
        <div 
          className={`border-2 border-dashed rounded-lg p-4 w-full max-w-2xl mx-auto flex flex-col items-center justify-center cursor-pointer hover:bg-gray-50 transition ${isLoading ? 'opacity-50' : ''}`}
          onClick={() => !isLoading && document.getElementById("group-photo").click()}
        >
          {groupImage ? (
            <img 
              src={groupImage} 
              alt="Group Preview" 
              className="max-h-96 object-contain"
            />
          ) : (
            <div className="text-center p-12">
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
          className="w-48 bg-blue-600 hover:bg-blue-700 text-white"
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
        <div className="space-y-6">
          <div className="border rounded-lg p-4 bg-white">
            <h3 className="text-lg font-semibold text-gray-700 mb-4">Detected Faces</h3>
            <div className="flex justify-center">
              <img 
                src={results.visualizedImage} 
                alt="Analyzed Group" 
                className="max-w-full max-h-[600px] object-contain"
              />
            </div>
          </div>

          <Alert className="bg-blue-50 border-blue-200">
            <AlertCircle className="h-4 w-4 text-blue-600" />
            <AlertDescription>
              <p className="font-semibold text-blue-700">
                Detected {results.totalFaces} faces in the image
              </p>
              <div className="text-sm text-gray-600 mt-1">
                <p>Successfully processed: {results.processedFaces} faces</p>
                <p>Processing Time: {results.processingTime.toFixed(2)}s</p>
              </div>
            </AlertDescription>
          </Alert>

          {results.results.length > 0 ? (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-700">
                Detected Relationships:
              </h3>
              {results.results.map((result, idx) => (
                <Alert key={idx} className="bg-green-50 border-green-200">
                  <AlertCircle className="h-4 w-4 text-green-600" />
                  <AlertDescription>
                    <div className="space-y-2">
                      <p className="font-semibold text-green-700">
                        Person {result.pair[0]} ({result.locations[0].gender}) and 
                        Person {result.pair[1]} ({result.locations[1].gender})
                      </p>
                      <div className="text-sm text-gray-600 space-y-1">
                        <p className="font-medium text-green-700">
                          Relationship: {result.relationship.description}
                        </p>
                        <p>Relationship Confidence: {(result.relationship.confidence * 100).toFixed(1)}%</p>
                        <p>Kinship Confidence: {(result.confidence * 100).toFixed(1)}%</p>
                        <p>Similarity Score: {result.similarity.toFixed(3)}</p>
                      </div>
                    </div>
                  </AlertDescription>
                </Alert>
              ))}
            </div>
          ) : (
            <Alert className="bg-yellow-50 border-yellow-200">
              <AlertCircle className="h-4 w-4 text-yellow-600" />
              <AlertDescription className="text-yellow-700">
                No kinship relationships detected between the faces.
              </AlertDescription>
            </Alert>
          )}
        </div>
      )}
    </div>
  );
};

const KinshipAnalyzer = () => {
  return (
    <Card className="w-full max-w-4xl mx-auto bg-white shadow-lg">
      <CardHeader className="border-b">
        <CardTitle className="text-2xl font-bold text-gray-900 text-center">
          Kinship Verification System
        </CardTitle>
      </CardHeader>
      <CardContent className="p-6">
        <Tabs defaultValue="pair" className="w-full">
          <TabsList className="grid w-full grid-cols-2 mb-8">
            <TabsTrigger 
              value="pair"
              className="data-[state=active]:bg-blue-100 data-[state=active]:text-blue-700"
            >
              Pair Comparison
            </TabsTrigger>
            <TabsTrigger 
              value="group"
              className="data-[state=active]:bg-blue-100 data-[state=active]:text-blue-700"
            >
              Group Analysis
            </TabsTrigger>
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