import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { User, Plane, MapPin, Award, Calendar, Clock, UserCheck, AlertTriangle } from "lucide-react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface CrewMember {
  Crew_ID: string;
  Name: string;
  Base: string;
  Rank: string;
  Qualification: string;
  Aircraft_Type_License: string;
  Leave_Start: string | null;
  Leave_End: string | null;
}

interface CrewSchedule {
  roster_id: number;
  start_date: string;
  end_date: string;
  violation_count: number;
}

interface CrewPreference {
  type: string;
  detail: string;
  priority: string;
}

const CrewList: React.FC = () => {
  const [crewMembers, setCrewMembers] = useState<CrewMember[]>([]);
  const [filteredCrew, setFilteredCrew] = useState<CrewMember[]>([]);
  const [selectedCrew, setSelectedCrew] = useState<CrewMember | null>(null);
  const [crewSchedule, setCrewSchedule] = useState<CrewSchedule[]>([]);
  const [crewPreferences, setCrewPreferences] = useState<CrewPreference[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [baseFilter, setBaseFilter] = useState('all');
  const [rankFilter, setRankFilter] = useState('all');
  const [aircraftFilter, setAircraftFilter] = useState('all');

  const baseOptions = ['DEL', 'BOM', 'BLR', 'MAA', 'CCU', 'HYD'];
  const rankOptions = ['Captain', 'First Officer', 'Purser', 'FA'];
  const aircraftOptions = ['A320neo', 'A321neo', 'ATR'];

  useEffect(() => {
    fetchCrewMembers();
  }, []);

  useEffect(() => {
    filterCrewMembers();
  }, [crewMembers, searchTerm, baseFilter, rankFilter, aircraftFilter]);

  const fetchCrewMembers = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://127.0.0.1:8000/api/v1/crew?limit=100&offset=0');
      const data = await response.json();
      setCrewMembers(data.crew_members || []);
    } catch (error) {
      console.error('Error fetching crew members:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchCrewSchedule = async (crewId: string) => {
    try {
      const response = await fetch(`http://127.0.0.1:8000/api/v1/crew/${crewId}/schedule`);
      const data = await response.json();
      setCrewSchedule(data.schedules || []);
    } catch (error) {
      console.error('Error fetching crew schedule:', error);
      setCrewSchedule([]);
    }
  };

  const fetchCrewPreferences = async (crewId: string) => {
    try {
      const response = await fetch(`http://127.0.0.1:8000/api/v1/preferences/${crewId}`);
      const data = await response.json();
      setCrewPreferences(data.preferences || []);
    } catch (error) {
      console.error('Error fetching crew preferences:', error);
      setCrewPreferences([]);
    }
  };

  const filterCrewMembers = () => {
    let filtered = crewMembers;

    if (searchTerm) {
      filtered = filtered.filter(crew => 
        crew.Name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        crew.Crew_ID.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    if (baseFilter && baseFilter !== 'all') {
      filtered = filtered.filter(crew => crew.Base === baseFilter);
    }

    if (rankFilter && rankFilter !== 'all') {
      filtered = filtered.filter(crew => crew.Rank === rankFilter);
    }

    if (aircraftFilter && aircraftFilter !== 'all') {
      filtered = filtered.filter(crew => 
        crew.Aircraft_Type_License.includes(aircraftFilter)
      );
    }

    setFilteredCrew(filtered);
  };

  const handleCrewSelect = async (crew: CrewMember) => {
    setSelectedCrew(crew);
    await Promise.all([
      fetchCrewSchedule(crew.Crew_ID),
      fetchCrewPreferences(crew.Crew_ID)
    ]);
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

  const getQualificationColor = (qualification: string) => {
    switch (qualification) {
      case 'Senior': return 'bg-primary-light text-primary-foreground';
      case 'Line Checked': return 'bg-success-light text-success-foreground';
      case 'Training Captain': return 'bg-warning-light text-warning-foreground';
      default: return 'bg-muted text-muted-foreground';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'High': return 'text-destructive';
      case 'Medium': return 'text-warning';
      case 'Low': return 'text-success';
      default: return 'text-muted-foreground';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Filter Controls */}
      <Card className="shadow-elegant">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <User className="h-5 w-5" />
            Crew Filters
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
            <Input
              placeholder="Search crew..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full"
            />
            <Select value={baseFilter} onValueChange={setBaseFilter}>
              <SelectTrigger>
                <SelectValue placeholder="Base" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Bases</SelectItem>
                {baseOptions.map(base => (
                  <SelectItem key={base} value={base}>{base}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Select value={rankFilter} onValueChange={setRankFilter}>
              <SelectTrigger>
                <SelectValue placeholder="Rank" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Ranks</SelectItem>
                {rankOptions.map(rank => (
                  <SelectItem key={rank} value={rank}>{rank}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Select value={aircraftFilter} onValueChange={setAircraftFilter}>
              <SelectTrigger>
                <SelectValue placeholder="Aircraft" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Aircraft</SelectItem>
                {aircraftOptions.map(aircraft => (
                  <SelectItem key={aircraft} value={aircraft}>{aircraft}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button 
              variant="outline" 
              onClick={() => {
                setSearchTerm('');
                setBaseFilter('all');
                setRankFilter('all');
                setAircraftFilter('all');
              }}
            >
              Clear Filters
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Crew List */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredCrew.map((crew) => (
          <Card key={crew.Crew_ID} className="hover:shadow-glow transition-all duration-300 cursor-pointer">
            <CardContent className="p-6">
              <div className="space-y-4">
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="font-semibold text-lg">{crew.Name}</h3>
                    <p className="text-muted-foreground">{crew.Crew_ID}</p>
                  </div>
                  <Badge className={getRankColor(crew.Rank)}>
                    {crew.Rank}
                  </Badge>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center gap-2 text-sm">
                    <MapPin className="h-4 w-4 text-muted-foreground" />
                    <span>Base: {crew.Base}</span>
                  </div>
                  <div className="flex items-center gap-2 text-sm">
                    <Award className="h-4 w-4 text-muted-foreground" />
                    <Badge variant="outline" className={getQualificationColor(crew.Qualification)}>
                      {crew.Qualification}
                    </Badge>
                  </div>
                  <div className="flex items-center gap-2 text-sm">
                    <Plane className="h-4 w-4 text-muted-foreground" />
                    <span className="text-xs">{crew.Aircraft_Type_License}</span>
                  </div>
                  {(crew.Leave_Start || crew.Leave_End) && (
                    <div className="flex items-center gap-2 text-sm">
                      <Calendar className="h-4 w-4 text-destructive" />
                      <span className="text-destructive">On Leave</span>
                    </div>
                  )}
                </div>

                <Dialog>
                  <DialogTrigger asChild>
                    <Button 
                      variant="outline" 
                      className="w-full"
                      onClick={() => handleCrewSelect(crew)}
                    >
                      View Details
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="max-w-4xl max-h-[80vh] overflow-auto">
                    <DialogHeader>
                      <DialogTitle className="flex items-center gap-2">
                        <User className="h-5 w-5" />
                        {selectedCrew?.Name} ({selectedCrew?.Crew_ID})
                      </DialogTitle>
                    </DialogHeader>
                    
                    {selectedCrew && (
                      <Tabs defaultValue="profile" className="w-full">
                        <TabsList className="grid w-full grid-cols-3">
                          <TabsTrigger value="profile">Profile</TabsTrigger>
                          <TabsTrigger value="schedule">Schedule</TabsTrigger>
                          <TabsTrigger value="preferences">Preferences</TabsTrigger>
                        </TabsList>
                        
                        <TabsContent value="profile" className="space-y-4">
                          <Card>
                            <CardHeader>
                              <CardTitle>Basic Information</CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-4">
                              <div className="grid grid-cols-2 gap-4">
                                <div>
                                  <label className="text-sm font-medium text-muted-foreground">Crew ID</label>
                                  <p className="font-mono">{selectedCrew.Crew_ID}</p>
                                </div>
                                <div>
                                  <label className="text-sm font-medium text-muted-foreground">Name</label>
                                  <p>{selectedCrew.Name}</p>
                                </div>
                                <div>
                                  <label className="text-sm font-medium text-muted-foreground">Base</label>
                                  <p>{selectedCrew.Base}</p>
                                </div>
                                <div>
                                  <label className="text-sm font-medium text-muted-foreground">Rank</label>
                                  <Badge className={getRankColor(selectedCrew.Rank)}>
                                    {selectedCrew.Rank}
                                  </Badge>
                                </div>
                                <div>
                                  <label className="text-sm font-medium text-muted-foreground">Qualification</label>
                                  <Badge variant="outline" className={getQualificationColor(selectedCrew.Qualification)}>
                                    {selectedCrew.Qualification}
                                  </Badge>
                                </div>
                                <div>
                                  <label className="text-sm font-medium text-muted-foreground">Aircraft License</label>
                                  <p className="text-sm">{selectedCrew.Aircraft_Type_License}</p>
                                </div>
                              </div>
                              
                              {(selectedCrew.Leave_Start || selectedCrew.Leave_End) && (
                                <div className="mt-4 p-4 bg-destructive/10 rounded-lg">
                                  <div className="flex items-center gap-2 text-destructive">
                                    <AlertTriangle className="h-4 w-4" />
                                    <span className="font-medium">Currently on Leave</span>
                                  </div>
                                  <p className="text-sm text-muted-foreground mt-1">
                                    {selectedCrew.Leave_Start} - {selectedCrew.Leave_End}
                                  </p>
                                </div>
                              )}
                            </CardContent>
                          </Card>
                        </TabsContent>
                        
                        <TabsContent value="schedule" className="space-y-4">
                          <Card>
                            <CardHeader>
                              <CardTitle>Recent Schedule History</CardTitle>
                            </CardHeader>
                            <CardContent>
                              {crewSchedule.length > 0 ? (
                                <div className="space-y-3">
                                  {crewSchedule.slice(0, 10).map((schedule, index) => (
                                    <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                                      <div className="space-y-1">
                                        <p className="font-medium">Roster #{schedule.roster_id}</p>
                                        <p className="text-sm text-muted-foreground">
                                          {schedule.start_date} - {schedule.end_date}
                                        </p>
                                      </div>
                                      <Badge variant={schedule.violation_count > 0 ? "destructive" : "secondary"}>
                                        {schedule.violation_count} violations
                                      </Badge>
                                    </div>
                                  ))}
                                </div>
                              ) : (
                                <p className="text-muted-foreground">No schedule history available</p>
                              )}
                            </CardContent>
                          </Card>
                        </TabsContent>
                        
                        <TabsContent value="preferences" className="space-y-4">
                          <Card>
                            <CardHeader>
                              <CardTitle>Crew Preferences</CardTitle>
                            </CardHeader>
                            <CardContent>
                              {crewPreferences.length > 0 ? (
                                <div className="space-y-3">
                                  {crewPreferences.map((pref, index) => (
                                    <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                                      <div className="space-y-1">
                                        <p className="font-medium">{pref.type.replace('_', ' ')}</p>
                                        <p className="text-sm text-muted-foreground">{pref.detail}</p>
                                      </div>
                                      <Badge variant="outline" className={getPriorityColor(pref.priority)}>
                                        {pref.priority} Priority
                                      </Badge>
                                    </div>
                                  ))}
                                </div>
                              ) : (
                                <p className="text-muted-foreground">No preferences set</p>
                              )}
                            </CardContent>
                          </Card>
                        </TabsContent>
                      </Tabs>
                    )}
                  </DialogContent>
                </Dialog>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {filteredCrew.length === 0 && (
        <Card>
          <CardContent className="text-center py-8">
            <User className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-muted-foreground">No crew members found matching your criteria</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default CrewList;