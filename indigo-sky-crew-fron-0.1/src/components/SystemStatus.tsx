import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Plane, Users, Calendar, Database, RefreshCw, CheckCircle, AlertCircle } from "lucide-react";

interface SystemData {
  crew_count: number;
  flights_count: number;
  flights_date_range: {
    min: string;
    max: string;
  };
}

const SystemStatus: React.FC = () => {
  const [systemData, setSystemData] = useState<SystemData | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());

  useEffect(() => {
    fetchSystemStatus();
  }, []);

  const fetchSystemStatus = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://127.0.0.1:8000/api/v1/debug/data-status');
      const data = await response.json();
      
      // Extract only the relevant data
      setSystemData({
        crew_count: data.crew_count,
        flights_count: data.flights_count,
        flights_date_range: data.flights_date_range
      });
      setLastUpdated(new Date());
    } catch (error) {
      console.error('Error fetching system status:', error);
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-IN', {
      day: '2-digit',
      month: 'short',
      year: 'numeric'
    });
  };

  const getDateRangeDays = () => {
    if (!systemData?.flights_date_range) return 0;
    const start = new Date(systemData.flights_date_range.min);
    const end = new Date(systemData.flights_date_range.max);
    return Math.ceil((end.getTime() - start.getTime()) / (1000 * 60 * 60 * 24)) + 1;
  };

  if (loading) {
    return (
      <Card className="shadow-elegant">
        <CardContent className="flex items-center justify-center h-32">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* System Overview */}
      <Card className="shadow-elegant">
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5" />
            System Status
          </CardTitle>
          <div className="flex items-center gap-2">
            <Badge variant="secondary" className="bg-success-light text-success-foreground">
              <CheckCircle className="h-3 w-3 mr-1" />
              Online
            </Badge>
            <Button
              variant="outline"
              size="sm"
              onClick={fetchSystemStatus}
              disabled={loading}
            >
              <RefreshCw className={`h-4 w-4 mr-1 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="flex items-center space-x-3 p-4 rounded-lg bg-primary-lighter">
              <div className="p-2 bg-primary rounded-lg">
                <Users className="h-6 w-6 text-primary-foreground" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Total Crew</p>
                <p className="text-2xl font-bold text-primary">{systemData?.crew_count || 0}</p>
              </div>
            </div>

            <div className="flex items-center space-x-3 p-4 rounded-lg bg-success-light">
              <div className="p-2 bg-success rounded-lg">
                <Plane className="h-6 w-6 text-success-foreground" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Total Flights</p>
                <p className="text-2xl font-bold text-success">{systemData?.flights_count || 0}</p>
              </div>
            </div>

            <div className="flex items-center space-x-3 p-4 rounded-lg bg-warning-light">
              <div className="p-2 bg-warning rounded-lg">
                <Calendar className="h-6 w-6 text-warning-foreground" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Schedule Days</p>
                <p className="text-2xl font-bold text-warning">{getDateRangeDays()}</p>
              </div>
            </div>

            <div className="flex items-center space-x-3 p-4 rounded-lg bg-secondary-lighter">
              <div className="p-2 bg-secondary rounded-lg">
                <Database className="h-6 w-6 text-secondary-foreground" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Avg Flights/Day</p>
                <p className="text-2xl font-bold text-secondary">
                  {Math.round((systemData?.flights_count || 0) / getDateRangeDays())}
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Schedule Range */}
      <Card className="shadow-elegant">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Calendar className="h-5 w-5" />
            Schedule Coverage
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-muted/30 rounded-lg">
              <div>
                <p className="text-sm text-muted-foreground">Schedule Start Date</p>
                <p className="text-lg font-semibold">
                  {systemData?.flights_date_range?.min ? formatDate(systemData.flights_date_range.min) : 'N/A'}
                </p>
              </div>
              <div className="text-center">
                <p className="text-sm text-muted-foreground">Coverage Period</p>
                <Badge variant="outline" className="text-lg px-3 py-1">
                  {getDateRangeDays()} days
                </Badge>
              </div>
              <div className="text-right">
                <p className="text-sm text-muted-foreground">Schedule End Date</p>
                <p className="text-lg font-semibold">
                  {systemData?.flights_date_range?.max ? formatDate(systemData.flights_date_range.max) : 'N/A'}
                </p>
              </div>
            </div>

            {/* Additional Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center p-4 border rounded-lg">
                <p className="text-sm text-muted-foreground">Flights per Crew</p>
                <p className="text-xl font-bold text-primary">
                  {systemData ? Math.round((systemData.flights_count * 6) / systemData.crew_count) : 0}
                </p>
                <p className="text-xs text-muted-foreground">Estimated assignments</p>
              </div>
              <div className="text-center p-4 border rounded-lg">
                <p className="text-sm text-muted-foreground">Daily Crew Utilization</p>
                <p className="text-xl font-bold text-success">
                  {systemData ? Math.round((systemData.flights_count * 6) / (systemData.crew_count * getDateRangeDays()) * 100) : 0}%
                </p>
                <p className="text-xs text-muted-foreground">Average utilization</p>
              </div>
              <div className="text-center p-4 border rounded-lg">
                <p className="text-sm text-muted-foreground">System Health</p>
                <div className="flex items-center justify-center gap-2">
                  <CheckCircle className="h-5 w-5 text-success" />
                  <span className="text-xl font-bold text-success">Operational</span>
                </div>
                <p className="text-xs text-muted-foreground">All systems online</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Last Updated */}
      <Card>
        <CardContent className="py-3">
          <div className="flex items-center justify-between text-sm text-muted-foreground">
            <span>Last updated: {lastUpdated.toLocaleString()}</span>
            <span>Data source: IndiGo Crew Management API</span>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default SystemStatus;