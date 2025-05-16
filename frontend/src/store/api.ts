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
  tagTypes: ['Analysis', 'User'],
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
} = api; 