import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';
import { RootState } from './index';

interface LoginRequest {
  username: string;
  password: string;
}

interface RegisterRequest {
  username: string;
  password: string;
}

interface User {
  username: string;
}

interface LoginResponse {
  access_token: string;
  token_type: string;
}

interface TextAnalysisRequest {
  text: string;
  features: string[];
  compare_text?: string;
}

interface TextAnalysisResponse {
  result: {
    entities: [string, string][];
    tokens: string[];
    pos_tags: [string, string][];
    sentiment?: {
      polarity: number;
      subjectivity: number;
    };
    keywords?: [string, number][];
    word_frequency?: Record<string, number>;
    dependencies?: [string, string, string][];
  };
}

interface ImageProcessingRequest {
  file: File;
  features: string[];
  brightness?: number;
  hue?: number;
  saturation?: number;
  rotation?: number;
  flip?: 'horizontal' | 'vertical';
}

interface ImageProcessingResult {
  id: string;
  image_url: string;
  result: {
    brightness?: string;
    hue?: string;
    saturation?: string;
    rotation?: string;
    flip?: string;
    enhanced?: string;
    detections?: Array<{
      class: string;
      confidence: number;
      bbox: number[];
    }>;
    segmentation?: {
      segments: number;
      mask: string;
      segmented: string;
      top_segments?: Array<{
        id: number;
        area: number;
        perimeter: number;
        bbox: number[];
        image: string;
      }>;
    };
    styled?: string;
  };
  timestamp: string;
  username: string;
}

export const api = createApi({
  baseQuery: fetchBaseQuery({
    baseUrl: 'http://localhost:8000/api',
    prepareHeaders: (headers, { getState }) => {
      const token = (getState() as RootState).auth.token;
      if (token) {
        headers.set('authorization', `Bearer ${token}`);
      }
      return headers;
    },
  }),
  tagTypes: ['Analysis', 'User', 'Image'],
  endpoints: (builder) => ({
    login: builder.mutation<LoginResponse, LoginRequest>({
      query: (credentials) => ({
        url: '/auth/login',
        method: 'POST',
        body: new URLSearchParams({
          username: credentials.username,
          password: credentials.password,
        }),
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
      }),
    }),
    register: builder.mutation<{ username: string }, RegisterRequest>({
      query: (credentials) => ({
        url: '/auth/register',
        method: 'POST',
        body: credentials,
      }),
    }),
    changePassword: builder.mutation<{ message: string }, { new_password: string }>({
      query: (data) => ({
        url: '/auth/change-password',
        method: 'POST',
        body: data,
      }),
      invalidatesTags: ['User'],
    }),
    getUserProfile: builder.query<{ username: string; created_at: string }, void>({
      query: () => '/users/profile',
      providesTags: ['User'],
    }),
    getUserStats: builder.query<{ today: number; week: number; total: number }, void>({
      query: () => '/users/stats',
      providesTags: ['Analysis'],
    }),
    analyzeText: builder.mutation<{
      id: string;
      text: string;
      result: TextAnalysisResponse['result'];
      timestamp: string;
      username: string;
    }, TextAnalysisRequest>({
      query: (data) => ({
        url: '/nlp/analyze',
        method: 'POST',
        body: data,
        headers: {
          'Content-Type': 'application/json',
        },
      }),
      transformErrorResponse: (response) => {
        console.error('Analysis error:', response);
        return response;
      },
      invalidatesTags: ['Analysis'],
    }),
    getAnalysisHistory: builder.query<Array<{
      id: string;
      text: string;
      result: any;
      timestamp: string;
      username: string;
    }>, void>({
      query: () => '/nlp/history',
      providesTags: ['Analysis'],
    }),
    processImage: builder.mutation<ImageProcessingResult, ImageProcessingRequest>({
      query: ({ file, features }) => {
        const formData = new FormData();
        formData.append('file', file);
        features.forEach(feature => formData.append('features', feature));
        
        return {
          url: '/image/process',
          method: 'POST',
          body: formData,
        };
      },
      invalidatesTags: ['Image'],
    }),
    getImageHistory: builder.query<ImageProcessingResult[], void>({
      query: () => '/image/history',
      providesTags: ['Image'],
    }),
  }),
});

export const {
  useLoginMutation,
  useRegisterMutation,
  useChangePasswordMutation,
  useGetUserProfileQuery,
  useGetUserStatsQuery,
  useAnalyzeTextMutation,
  useGetAnalysisHistoryQuery,
  useProcessImageMutation,
  useGetImageHistoryQuery,
} = api; 