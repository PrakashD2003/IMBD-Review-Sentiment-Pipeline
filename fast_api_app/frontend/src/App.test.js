import { render, screen } from '@testing-library/react';
import App from './App';

test('renders navigation links', () => {
  render(<App />);
  const predictLink = screen.getByRole('link', { name: /Predict/i });
  const trainLink = screen.getByRole('link', { name: /Train/i });
  expect(predictLink).toBeInTheDocument();
  expect(trainLink).toBeInTheDocument();
});
