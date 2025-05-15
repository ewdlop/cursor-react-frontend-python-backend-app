import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface AnalysisResult {
  id: string;
  text: string;
  result: any;
  timestamp: string;
}

interface AnalysisState {
  history: AnalysisResult[];
  stats: {
    today: number;
    week: number;
    total: number;
  };
}

const initialState: AnalysisState = {
  history: [],
  stats: {
    today: 0,
    week: 0,
    total: 0,
  },
};

const analysisSlice = createSlice({
  name: 'analysis',
  initialState,
  reducers: {
    addAnalysis: (state, action: PayloadAction<AnalysisResult>) => {
      state.history.unshift(action.payload);
      state.stats.total += 1;
      state.stats.today += 1;
      state.stats.week += 1;
    },
    clearHistory: (state) => {
      state.history = [];
    },
    updateStats: (state, action: PayloadAction<{ today: number; week: number; total: number }>) => {
      state.stats = action.payload;
    },
  },
});

export const { addAnalysis, clearHistory, updateStats } = analysisSlice.actions;
export default analysisSlice.reducer; 