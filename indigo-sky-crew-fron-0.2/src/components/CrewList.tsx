import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { 
  Users, 
  Search, 
  Filter, 
  MapPin, 
  Award, 
  Plane,
  RefreshCw,
  X
} from "lucide-react";
import { toast } from "sonner";
import CrewDetailsModal from './CrewDetailsModal';

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

const CrewList = () => {
  const [crewMembers, setCrewMembers] = useState<CrewMember[]>([]);
  const [filteredCrew, setFilteredCrew] = useState<CrewMember[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedBase, setSelectedBase] = useState('all');
  const [selectedRank, setSelectedRank] = useState('all');
  const [selectedAircraft, setSelectedAircraft] = useState('all');
  const [selectedCrewId, setSelectedCrewId] = useState<string | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  useEffect(() => {
    fetchCrewMembers();
  }, []);

  useEffect(() => {
    filterCrewMembers();
  }, [crewMembers, searchTerm, selectedBase, selectedRank, selectedAircraft]);

  const fetchCrewMembers = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://127.0.0.1:8000/api/v1/crew?limit=100&offset=0');
      const data = await response.json();
      setCrewMembers(data.crew_members || []);
    } catch (error) {
      console.error('Error fetching crew members:', error);
      toast.error('Failed to fetch crew members');
    } finally {
      setLoading(false);
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

    if (selectedBase !== 'all') {
      filtered = filtered.filter(crew => crew.Base === selectedBase);
    }

    if (selectedRank !== 'all') {
      filtered = filtered.filter(crew => crew.Rank === selectedRank);
    }

    if (selectedAircraft !== 'all') {
      filtered = filtered.filter(crew => 
        crew.Aircraft_Type_License.includes(selectedAircraft)
      );
    }

    setFilteredCrew(filtered);
  };

  const clearFilters = () => {
    setSearchTerm('');
    setSelectedBase('all');
    setSelectedRank('all');
    setSelectedAircraft('all');
  };

  const handleCrewClick = (crewId: string) => {
    setSelectedCrewId(crewId);
    setIsModalOpen(true);
  };

  const getRankColor = (rank: string) => {
    switch (rank.toLowerCase()) {
      case 'captain': return 'bg-primary text-primary-foreground';
      case 'first officer': return 'bg-emerald-500 text-white';
      case 'senior first officer': return 'bg-emerald-600 text-white';
      case 'instructor/check pilot': return 'bg-purple-600 text-white';
      case 'purser': return 'bg-amber-500 text-white';
      case 'senior cabin crew': return 'bg-blue-500 text-white';
      case 'junior cabin crew': return 'bg-gray-500 text-white';
      case 'sccm (lead cabin crew)': return 'bg-indigo-600 text-white';
      case 'trainee cabin crew': return 'bg-orange-500 text-white';
      default: return 'bg-secondary text-secondary-foreground';
    }
  };

  const getQualificationColor = (qualification: string) => {
    switch (qualification.toLowerCase()) {
      case 'line checked': return 'text-green-600';
      case 'base check': return 'text-blue-600';
      case 'senior': return 'text-purple-600';
      case 'junior': return 'text-orange-600';
      default: return 'text-muted-foreground';
    }
  };

  const getUniqueValues = (field: keyof CrewMember) => {
    const values = crewMembers.map(crew => crew[field]).filter(Boolean);
    return [...new Set(values)];
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col lg:flex-row gap-4 items-start lg:items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-foreground">Crew Management</h2>
          <p className="text-muted-foreground">Manage and view crew member information</p>
        </div>
        <Button onClick={fetchCrewMembers} variant="outline" className="gap-2">
          <RefreshCw className="h-4 w-4" />
          Refresh
        </Button>
      </div>

      {/* Filters */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Filter className="h-5 w-5" />
            Filters
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
            <div className="relative">
              <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search crew..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10"
              />
            </div>
            
            <Select value={selectedBase} onValueChange={setSelectedBase}>
              <SelectTrigger>
                <SelectValue placeholder="Select Base" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Bases</SelectItem>
                {getUniqueValues('Base').map(base => (
                  <SelectItem key={base} value={base}>{base}</SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Select value={selectedRank} onValueChange={setSelectedRank}>
              <SelectTrigger>
                <SelectValue placeholder="Select Rank" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Ranks</SelectItem>
                {getUniqueValues('Rank').map(rank => (
                  <SelectItem key={rank} value={rank}>{rank}</SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Select value={selectedAircraft} onValueChange={setSelectedAircraft}>
              <SelectTrigger>
                <SelectValue placeholder="Aircraft Type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Aircraft</SelectItem>
                <SelectItem value="A320neo">A320neo</SelectItem>
                <SelectItem value="A321neo">A321neo</SelectItem>
              </SelectContent>
            </Select>

            <Button variant="outline" onClick={clearFilters} className="gap-2">
              <X className="h-4 w-4" />
              Clear
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Results Summary */}
      <div className="flex items-center gap-4 text-sm text-muted-foreground">
        <span>Showing {filteredCrew.length} of {crewMembers.length} crew members</span>
      </div>

      {/* Crew Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredCrew.map((crew, index) => (
          <Card key={`${crew.Crew_ID}-${index}`} className="hover:shadow-lg transition-all duration-300 cursor-pointer group">
            <CardContent className="p-6">
              <div className="space-y-4">
                <div className="flex items-start justify-between">
                  <div>
                    <button 
                      onClick={() => handleCrewClick(crew.Crew_ID)}
                      className="text-left hover:text-primary transition-colors"
                    >
                      <h3 className="font-semibold text-lg hover:underline">{crew.Name}</h3>
                      <p className="text-sm text-muted-foreground hover:text-primary/70">{crew.Crew_ID}</p>
                    </button>
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
                    <span className={getQualificationColor(crew.Qualification)}>
                      {crew.Qualification}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 text-sm">
                    <Plane className="h-4 w-4 text-muted-foreground" />
                    <span className="text-xs">{crew.Aircraft_Type_License}</span>
                  </div>
                  {(crew.Leave_Start || crew.Leave_End) && (
                    <div className="flex items-center gap-2 text-sm">
                      <div className="h-2 w-2 rounded-full bg-red-500" />
                      <span className="text-red-600">On Leave: {crew.Leave_Start} - {crew.Leave_End}</span>
                    </div>
                  )}
                </div>

                <Button 
                  variant="outline" 
                  className="w-full group-hover:bg-primary group-hover:text-primary-foreground transition-colors"
                  onClick={() => handleCrewClick(crew.Crew_ID)}
                >
                  <Users className="h-4 w-4 mr-2" />
                  View Details
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {filteredCrew.length === 0 && !loading && (
        <Card>
          <CardContent className="flex flex-col items-center justify-center h-64">
            <Users className="h-16 w-16 text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium mb-2">No crew members found</h3>
            <p className="text-muted-foreground text-center">
              Try adjusting your filters or search criteria
            </p>
          </CardContent>
        </Card>
      )}

      <CrewDetailsModal 
        crewId={selectedCrewId}
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
      />
    </div>
  );
};

export default CrewList;