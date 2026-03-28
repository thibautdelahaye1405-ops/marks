import { Component, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  error: Error | null;
}

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  render() {
    if (this.state.error) {
      return (
        this.props.fallback ?? (
          <div style={{ padding: 20, color: "#f87171" }}>
            <div style={{ fontWeight: 600, marginBottom: 8 }}>Component error</div>
            <pre style={{ fontSize: 11, color: "#94a3b8", whiteSpace: "pre-wrap" }}>
              {this.state.error.message}
            </pre>
          </div>
        )
      );
    }
    return this.props.children;
  }
}
