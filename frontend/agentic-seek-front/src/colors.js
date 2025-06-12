export const colors = {
  primary: '#6366f1',
  primaryLight: '#818cf8',
  primaryDark: '#4f46e5',
  
  secondary: '#64748b',
  secondaryLight: '#94a3b8',
  secondaryDark: '#475569',
  
  accent: '#06d6a0',
  accentLight: '#34d399',
  accentDark: '#059669',
  
  success: '#22c55e',
  warning: '#f59e0b',
  error: '#ef4444',
  info: '#3b82f6',
  
  darkBackground: '#0f172a',
  darkBackgroundSecondary: '#1e293b',
  darkCard: '#1e2a3a',
  darkCardHover: '#2a3441',
  darkBorder: '#334155',
  darkText: '#f8fafc',
  darkTextSecondary: '#cbd5e1',
  darkTextMuted: '#94a3b8',
  
  lightBackground: '#ffffff',
  lightCard: '#ffffff',
  lightBorder: '#e2e8f0',
  lightText: '#1e293b',
  lightTextSecondary: '#475569',
  
  white: '#ffffff',
  black: '#000000',
  gray100: '#f1f5f9',
  gray200: '#e2e8f0',
  gray300: '#cbd5e1',
  gray400: '#94a3b8',
  gray500: '#64748b',
  gray600: '#475569',
  gray700: '#334155',
  gray800: '#1e293b',
  gray900: '#0f172a',
  
  agentUser: '#6366f1',
  agentSystem: '#06d6a0',
  agentBrowser: '#f59e0b',
  agentFile: '#8b5cf6',
  agentPlanner: '#3b82f6',
  agentMCP: '#ef4444',
  
  shadow: 'rgba(0, 0, 0, 0.1)',
  shadowDark: 'rgba(0, 0, 0, 0.3)',
  overlay: 'rgba(0, 0, 0, 0.5)',
};

export const getAgentColor = (agentName) => {
  const agentColors = {
    'Browser': colors.agentBrowser,
    'File Agent': colors.agentFile,
    'Planner': colors.agentPlanner,
    'MCP Agent': colors.agentMCP,
    'System': colors.agentSystem,
  };
  return agentColors[agentName] || colors.primary;
};
