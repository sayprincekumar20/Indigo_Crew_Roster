import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Calendar, dateFnsLocalizer } from 'react-big-calendar';
import { format, parse, startOfWeek, getDay } from 'date-fns';
import { enUS } from 'date-fns/locale';
import 'react-big-calendar/lib/css/react-big-calendar.css';
import { 
  Plane, 
  Users, 
  AlertTriangle, 
  Calendar as CalendarIcon, 
  Clock, 
  TrendingUp,
  MapPin,
  User,
  Zap,
  BarChart3,
  RefreshCw,
  Target,
  Award,
  Activity
} from "lucide-react";
import { toast } from "sonner";
import CrewList from "@/components/CrewList";
import FlightsList from "@/components/FlightsList";
import SystemStatus from "@/components/SystemStatus";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

const locales = {
  'en-US': enUS,
};

const localizer = dateFnsLocalizer({
  format,
  parse,
  startOfWeek,
  getDay,
  locales,
});

interface CrewMember {
  Crew_ID: string;
  Crew_Rank: string;
}

interface FlightData {
  Date: string;
  Flight_Number: string;
  Duty_Start: string;
  Duty_End: string;
  Aircraft_Type: string;
  Origin: string;
  Destination: string;
  Duration: string;
  Crew_Members: CrewMember[];
}

interface Violation {
  type: string;
  category: string;
  message: string;
}

interface OptimizationMetrics {
  total_assignments: number;
  crew_utilization: number;
  violation_count: number;
  fitness_score: number;
  fairness_score: number;
  max_duty_hours: number;
  min_duty_hours: number;
  avg_duty_hours: number;
  std_dev_duty_hours: number;
}

interface RosterResponse {
  roster: FlightData[];
  fitness_score: number;
  violations: Violation[];
  optimization_metrics: OptimizationMetrics;
}

interface RosterHistoryItem {
  id: number;
  created_at: string;
  start_date: string;
  end_date: string;
  fitness_score: number;
  violation_count: number;
}

const Index = () => {
  const [activeTab, setActiveTab] = useState("roster");
  const [rosterData, setRosterData] = useState<RosterResponse | null>(null);
  const [rosterHistory, setRosterHistory] = useState<RosterHistoryItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [startDate, setStartDate] = useState("2023-10-01");
  const [endDate, setEndDate] = useState("2023-10-01");

  // Calendar events
  const [events, setEvents] = useState<any[]>([]);

  useEffect(() => {
    fetchRosterHistory();
  }, []);

  useEffect(() => {
    if (rosterData?.roster) {
      const calendarEvents = rosterData.roster.map((flight, index) => ({
        id: index,
        title: `${flight.Flight_Number} | ${flight.Origin} → ${flight.Destination}`,
        start: new Date(flight.Duty_Start),
        end: new Date(flight.Duty_End),
        resource: flight,
      }));
      setEvents(calendarEvents);
    }
  }, [rosterData]);

  const fetchRosterHistory = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/api/v1/rosters/history?limit=20&offset=0');
      const data = await response.json();
      setRosterHistory(data.rosters || []);
    } catch (error) {
      console.error('Error fetching roster history:', error);
      toast.error('Failed to fetch roster history');
    }
  };

  const generateRoster = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:8000/api/v1/generate-roster', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          start_date: startDate,
          end_date: endDate,
          optimization_weights: {
            crew_utilization: 1.0,
            violation_penalty: 2.0,
            fairness: 0.5
          }
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate roster');
      }

      const data: RosterResponse = await response.json();
      setRosterData(data);
      toast.success('Roster generated successfully!');
      
      // Refresh history
      await fetchRosterHistory();
    } catch (error) {
      console.error('Error generating roster:', error);
      toast.error('Failed to generate roster');
    } finally {
      setLoading(false);
    }
  };

  const deleteRoster = async (rosterId: number) => {
    try {
      const response = await fetch(`http://127.0.0.1:8000/api/v1/rosters/${rosterId}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error('Failed to delete roster');
      }

      toast.success('Roster deleted successfully');
      await fetchRosterHistory();
    } catch (error) {
      console.error('Error deleting roster:', error);
      toast.error('Failed to delete roster');
    }
  };

  const getRankColor = (rank: string) => {
    switch (rank) {
      case 'Captain': return 'bg-primary text-primary-foreground';
      case 'First Officer': return 'bg-success text-success-foreground';
      case 'Purser': return 'bg-warning text-warning-foreground';
      case 'FA': return 'bg-secondary text-secondary-foreground';
      default: return 'bg-muted text-muted-foreground';
    }
  };

  const formatDateTime = (dateTime: string) => {
    return new Date(dateTime).toLocaleString('en-IN', {
      day: '2-digit',
      month: 'short',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getViolationsByCategory = () => {
    if (!rosterData?.violations) return [];
    
    const categories = rosterData.violations.reduce((acc, violation) => {
      acc[violation.category] = (acc[violation.category] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    return Object.entries(categories).map(([name, value]) => ({ name, value }));
  };

  const getDutyHoursData = () => {
    if (!rosterData?.roster) return [];
    
    const dutyHours = rosterData.roster.map(flight => {
      const start = new Date(flight.Duty_Start);
      const end = new Date(flight.Duty_End);
      const hours = (end.getTime() - start.getTime()) / (1000 * 60 * 60);
      return {
        flight: flight.Flight_Number,
        hours: Math.round(hours * 100) / 100,
        route: `${flight.Origin}-${flight.Destination}`
      };
    });

    return dutyHours.slice(0, 10); // Show top 10
  };

  const COLORS = ['#002F6C', '#FDB913', '#10B981', '#F59E0B', '#EF4444'];

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-lighter via-background to-secondary-lighter">
      <div className="container mx-auto p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-bold bg-gradient-primary bg-clip-text text-transparent">
              IndiGo Crew Roster
            </h1>
            <p className="text-muted-foreground mt-2">
              AI-Powered Crew Scheduling & Optimization System
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="secondary" className="bg-success-light text-success-foreground">
              <Zap className="h-3 w-3 mr-1" />
              Optimized
            </Badge>
          </div>
        </div>

        {/* System Status */}
        <SystemStatus />

        {/* Main Content */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-6 bg-card shadow-elegant">
            <TabsTrigger value="roster" className="flex items-center gap-2">
              <CalendarIcon className="h-4 w-4" />
              Roster
            </TabsTrigger>
            <TabsTrigger value="crew" className="flex items-center gap-2">
              <Users className="h-4 w-4" />
              Crew
            </TabsTrigger>
            <TabsTrigger value="flights" className="flex items-center gap-2">
              <Plane className="h-4 w-4" />
              Flights
            </TabsTrigger>
            <TabsTrigger value="conflicts" className="flex items-center gap-2">
              <AlertTriangle className="h-4 w-4" />
              Violations
            </TabsTrigger>
            <TabsTrigger value="analytics" className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              Analytics
            </TabsTrigger>
            <TabsTrigger value="history" className="flex items-center gap-2">
              <Clock className="h-4 w-4" />
              History
            </TabsTrigger>
          </TabsList>

          <TabsContent value="roster" className="space-y-6">
            {/* Roster Generation */}
            <Card className="shadow-elegant">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="h-5 w-5" />
                  Generate Optimized Roster
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                  <div className="space-y-2">
                    <Label htmlFor="start-date">Start Date</Label>
                    <Input
                      id="start-date"
                      type="date"
                      value={startDate}
                      onChange={(e) => setStartDate(e.target.value)}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="end-date">End Date</Label>
                    <Input
                      id="end-date"
                      type="date"
                      value={endDate}
                      onChange={(e) => setEndDate(e.target.value)}
                    />
                  </div>
                  <div className="flex items-end">
                    <Button 
                      onClick={generateRoster} 
                      disabled={loading}
                      className="w-full bg-gradient-primary hover:opacity-90"
                    >
                      {loading ? (
                        <>
                          <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                          Generating...
                        </>
                      ) : (
                        <>
                          <Zap className="mr-2 h-4 w-4" />
                          Generate Roster
                        </>
                      )}
                    </Button>
                  </div>
                </div>

                {rosterData && (
                  <div className="space-y-6">
                    {/* Metrics Overview */}
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                      <div className="flex items-center space-x-3 p-4 rounded-lg bg-primary-lighter">
                        <div className="p-2 bg-primary rounded-lg">
                          <Award className="h-6 w-6 text-primary-foreground" />
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">Fitness Score</p>
                          <p className="text-2xl font-bold text-primary">
                            {Math.round(rosterData.fitness_score)}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-3 p-4 rounded-lg bg-success-light">
                        <div className="p-2 bg-success rounded-lg">
                          <Users className="h-6 w-6 text-success-foreground" />
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">Utilization</p>
                          <p className="text-2xl font-bold text-success">
                            {rosterData.optimization_metrics.crew_utilization}%
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-3 p-4 rounded-lg bg-warning-light">
                        <div className="p-2 bg-warning rounded-lg">
                          <AlertTriangle className="h-6 w-6 text-warning-foreground" />
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">Violations</p>
                          <p className="text-2xl font-bold text-warning">
                            {rosterData.violations.length}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-3 p-4 rounded-lg bg-secondary-lighter">
                        <div className="p-2 bg-secondary rounded-lg">
                          <Activity className="h-6 w-6 text-secondary-foreground" />
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">Assignments</p>
                          <p className="text-2xl font-bold text-secondary">
                            {rosterData.optimization_metrics.total_assignments}
                          </p>
                        </div>
                      </div>
                    </div>

                    {/* Calendar View */}
                    <Card>
                      <CardHeader>
                        <CardTitle>Flight Schedule Calendar</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div style={{ height: '600px' }}>
                          <Calendar
                            localizer={localizer}
                            events={events}
                            startAccessor="start"
                            endAccessor="end"
                            style={{ height: '100%' }}
                            views={['month', 'week', 'day']}
                            defaultView="day"
                            step={60}
                            showMultiDayTimes
                            eventPropGetter={(event) => ({
                              style: {
                                backgroundColor: '#002F6C',
                                borderColor: '#FDB913',
                                color: 'white',
                              },
                            })}
                          />
                        </div>
                      </CardContent>
                    </Card>

                    {/* Flight Details */}
                    <Card>
                      <CardHeader>
                        <CardTitle>Flight Details</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-4">
                          {rosterData.roster.map((flight, index) => (
                            <div key={index} className="p-4 border rounded-lg space-y-3">
                              <div className="flex items-center justify-between">
                                <div className="flex items-center gap-3">
                                  <Badge variant="outline" className="font-mono">
                                    {flight.Flight_Number}
                                  </Badge>
                                  <div className="flex items-center gap-2">
                                    <MapPin className="h-4 w-4 text-muted-foreground" />
                                    <span>{flight.Origin} → {flight.Destination}</span>
                                  </div>
                                  <Badge className="bg-success text-success-foreground">
                                    {flight.Aircraft_Type}
                                  </Badge>
                                </div>
                                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                                  <Clock className="h-4 w-4" />
                                  {formatDateTime(flight.Duty_Start)} - {formatDateTime(flight.Duty_End)}
                                </div>
                              </div>
                              
                              <div className="flex flex-wrap gap-2">
                                {flight.Crew_Members.map((crew, crewIndex) => (
                                  <Badge key={crewIndex} className={getRankColor(crew.Crew_Rank)}>
                                    {crew.Crew_Rank}: {crew.Crew_ID}
                                  </Badge>
                                ))}
                              </div>
                            </div>
                          ))}
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="crew" className="space-y-6">
            <CrewList />
          </TabsContent>

          <TabsContent value="flights" className="space-y-6">
            <FlightsList />
          </TabsContent>

          <TabsContent value="conflicts" className="space-y-6">
            <Card className="shadow-elegant">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5" />
                  DGCA Compliance Violations
                </CardTitle>
              </CardHeader>
              <CardContent>
                {rosterData?.violations && rosterData.violations.length > 0 ? (
                  <div className="space-y-3">
                    {rosterData.violations.map((violation, index) => (
                      <div key={index} className="p-4 border-l-4 border-destructive bg-destructive/5 rounded-r-lg">
                        <div className="flex items-start gap-3">
                          <AlertTriangle className="h-5 w-5 text-destructive mt-0.5" />
                          <div className="flex-1">
                            <div className="flex items-center gap-2 mb-1">
                              <Badge variant="destructive">{violation.type}</Badge>
                              <Badge variant="outline">{violation.category}</Badge>
                            </div>
                            <p className="text-sm text-foreground">{violation.message}</p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <AlertTriangle className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                    <p className="text-muted-foreground">
                      {rosterData ? 'No violations found! ✅' : 'Generate a roster to see violations.'}
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="analytics" className="space-y-6">
            {rosterData ? (
              <>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Violations by Category Chart */}
                  <Card className="shadow-elegant">
                    <CardHeader>
                      <CardTitle>Violations by Category</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={300}>
                        <PieChart>
                          <Pie
                            data={getViolationsByCategory()}
                            cx="50%"
                            cy="50%"
                            labelLine={false}
                            label={({ name, value }) => `${name}: ${value}`}
                            outerRadius={80}
                            fill="#8884d8"
                            dataKey="value"
                          >
                            {getViolationsByCategory().map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                            ))}
                          </Pie>
                          <Tooltip />
                        </PieChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>

                  {/* Duty Hours Distribution */}
                  <Card className="shadow-elegant">
                    <CardHeader>
                      <CardTitle>Flight Duty Hours</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={getDutyHoursData()}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="flight" />
                          <YAxis />
                          <Tooltip />
                          <Bar dataKey="hours" fill="#002F6C" />
                        </BarChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>
                </div>

                {/* Detailed Metrics */}
                <Card className="shadow-elegant">
                  <CardHeader>
                    <CardTitle>Optimization Metrics</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                      <div className="text-center">
                        <p className="text-2xl font-bold text-primary">
                          {rosterData.optimization_metrics.fairness_score.toFixed(2)}
                        </p>
                        <p className="text-sm text-muted-foreground">Fairness Score</p>
                      </div>
                      <div className="text-center">
                        <p className="text-2xl font-bold text-success">
                          {rosterData.optimization_metrics.max_duty_hours.toFixed(1)}h
                        </p>
                        <p className="text-sm text-muted-foreground">Max Duty Hours</p>
                      </div>
                      <div className="text-center">
                        <p className="text-2xl font-bold text-warning">
                          {rosterData.optimization_metrics.min_duty_hours.toFixed(1)}h
                        </p>
                        <p className="text-sm text-muted-foreground">Min Duty Hours</p>
                      </div>
                      <div className="text-center">
                        <p className="text-2xl font-bold text-secondary">
                          {rosterData.optimization_metrics.avg_duty_hours.toFixed(1)}h
                        </p>
                        <p className="text-sm text-muted-foreground">Avg Duty Hours</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </>
            ) : (
              <Card>
                <CardContent className="text-center py-8">
                  <BarChart3 className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <p className="text-muted-foreground">Generate a roster to view analytics.</p>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="history" className="space-y-6">
            <Card className="shadow-elegant">
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="flex items-center gap-2">
                  <Clock className="h-5 w-5" />
                  Roster Generation History
                </CardTitle>
                <Button variant="outline" onClick={fetchRosterHistory}>
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Refresh
                </Button>
              </CardHeader>
              <CardContent>
                {rosterHistory.length > 0 ? (
                  <div className="space-y-3">
                    {rosterHistory.map((roster) => (
                      <div key={roster.id} className="flex items-center justify-between p-4 border rounded-lg">
                        <div className="space-y-1">
                          <div className="flex items-center gap-2">
                            <Badge variant="outline">#{roster.id}</Badge>
                            <span className="text-sm text-muted-foreground">
                              {new Date(roster.created_at).toLocaleString()}
                            </span>
                          </div>
                          <p className="text-sm">
                            Period: {roster.start_date} to {roster.end_date}
                          </p>
                        </div>
                        <div className="flex items-center gap-3">
                          <div className="text-right">
                            <p className="text-sm text-muted-foreground">Fitness: {Math.round(roster.fitness_score)}</p>
                            <Badge variant={roster.violation_count > 0 ? "destructive" : "secondary"}>
                              {roster.violation_count} violations
                            </Badge>
                          </div>
                          <Button
                            variant="destructive"
                            size="sm"
                            onClick={() => deleteRoster(roster.id)}
                          >
                            Delete
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <Clock className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                    <p className="text-muted-foreground">No roster history available.</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default Index;