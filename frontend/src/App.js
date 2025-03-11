import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import UserForm from './components/UserForm';
import WorkoutPlan from './components/WorkoutPlan';
import './App.css';

function App() {
  const [workoutPlan, setWorkoutPlan] = useState(null);

  return (
    <Router>
      <div className="app-container">
        <header>
          <h1>Fitness Recommendation Tool</h1>
        </header>
        <main>
          <Routes>
            <Route path="/" element={<UserForm setWorkoutPlan={setWorkoutPlan} />} />
            <Route path="/workout-plan" element={<WorkoutPlan workoutPlan={workoutPlan} />} />
          </Routes>
        </main>
        <footer>
          <p>&copy; {new Date().getFullYear()} Fitness Recommendation Tool</p>
        </footer>
      </div>
    </Router>
  );
}

export default App;