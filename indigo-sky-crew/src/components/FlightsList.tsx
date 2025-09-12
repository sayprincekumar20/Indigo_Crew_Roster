import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Plane, MapPin, Clock, Calendar, ArrowRight } from "lucide-react";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

interface Flight {
  Date: string;
  Flight_Number: string;
  Origin: string;
  Destination: string;
  Scheduled_Departure_UTC: string;
  Scheduled_Arrival_UTC: string;
  Aircraft_Type: string;
  Duration_HH_MM: string;
}

const FlightsList: React.FC = () => {
  const [flights, setFlights] = useState<Flight[]>([]);
  const [filteredFlights, setFilteredFlights] = useState<Flight[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [originFilter, setOriginFilter] = useState('all');
  const [destinationFilter, setDestinationFilter] = useState('all');
  const [aircraftFilter, setAircraftFilter] = useState('all');
  const [dateFilter, setDateFilter] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalFlights, setTotalFlights] = useState(0);
  const flightsPerPage = 20;

  const originOptions = ['DEL', 'BOM', 'BLR', 'MAA', 'CCU', 'HYD', 'GOI'];
  const destinationOptions = ['DEL', 'BOM', 'BLR', 'MAA', 'CCU', 'HYD', 'GOI'];
  const aircraftOptions = ['A320neo', 'A321neo', 'ATR'];

  useEffect(() => {
    fetchFlights();
  }, [currentPage, originFilter, destinationFilter, aircraftFilter, dateFilter]);

  useEffect(() => {
    filterFlights();
  }, [flights, searchTerm]);

  const fetchFlights = async () => {
    try {
      setLoading(true);
      const params = new URLSearchParams({
        limit: flightsPerPage.toString(),
        offset: ((currentPage - 1) * flightsPerPage).toString(),
      });

      if (originFilter && originFilter !== 'all') params.append('origin', originFilter);
      if (destinationFilter && destinationFilter !== 'all') params.append('destination', destinationFilter);
      if (aircraftFilter && aircraftFilter !== 'all') params.append('aircraft_type', aircraftFilter);
      if (dateFilter) params.append('date', dateFilter);

      const response = await fetch(`http://127.0.0.1:8000/api/v1/flights?${params}`);
      const data = await response.json();
      setFlights(data.flights || []);
      setTotalFlights(data.total || 0);
    } catch (error) {
      console.error('Error fetching flights:', error);
    } finally {
      setLoading(false);
    }
  };

  const filterFlights = () => {
    let filtered = flights;

    if (searchTerm) {
      filtered = filtered.filter(flight => 
        flight.Flight_Number.toLowerCase().includes(searchTerm.toLowerCase()) ||
        flight.Origin.toLowerCase().includes(searchTerm.toLowerCase()) ||
        flight.Destination.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    setFilteredFlights(filtered);
  };

  const getAircraftColor = (aircraft: string) => {
    switch (aircraft) {
      case 'A320neo': return 'bg-primary text-primary-foreground';
      case 'A321neo': return 'bg-success text-success-foreground';
      case 'ATR': return 'bg-warning text-warning-foreground';
      default: return 'bg-muted text-muted-foreground';
    }
  };

  const formatTime = (time: string) => {
    return time.substring(0, 5); // Remove seconds
  };

  const totalPages = Math.ceil(totalFlights / flightsPerPage);

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
            <Plane className="h-5 w-5" />
            Flight Filters
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-4">
            <Input
              placeholder="Search flights..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full"
            />
            <Input
              type="date"
              value={dateFilter}
              onChange={(e) => setDateFilter(e.target.value)}
              className="w-full"
            />
            <Select value={originFilter} onValueChange={setOriginFilter}>
              <SelectTrigger>
                <SelectValue placeholder="Origin" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Origins</SelectItem>
                {originOptions.map(origin => (
                  <SelectItem key={origin} value={origin}>{origin}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Select value={destinationFilter} onValueChange={setDestinationFilter}>
              <SelectTrigger>
                <SelectValue placeholder="Destination" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Destinations</SelectItem>
                {destinationOptions.map(dest => (
                  <SelectItem key={dest} value={dest}>{dest}</SelectItem>
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
                setOriginFilter('all');
                setDestinationFilter('all');
                setAircraftFilter('all');
                setDateFilter('');
                setCurrentPage(1);
              }}
            >
              Clear Filters
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Flights Summary */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <Plane className="h-4 w-4 text-primary" />
              <div>
                <p className="text-sm text-muted-foreground">Total Flights</p>
                <p className="text-xl font-bold text-primary">{totalFlights}</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <MapPin className="h-4 w-4 text-success" />
              <div>
                <p className="text-sm text-muted-foreground">Unique Routes</p>
                <p className="text-xl font-bold text-success">
                  {new Set(flights.map(f => `${f.Origin}-${f.Destination}`)).size}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <Calendar className="h-4 w-4 text-warning" />
              <div>
                <p className="text-sm text-muted-foreground">Aircraft Types</p>
                <p className="text-xl font-bold text-warning">
                  {new Set(flights.map(f => f.Aircraft_Type)).size}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4 text-secondary" />
              <div>
                <p className="text-sm text-muted-foreground">Avg Duration</p>
                <p className="text-xl font-bold text-secondary">
                  {flights.length > 0 ? 
                    Math.round(flights.reduce((acc, f) => {
                      const [hours, minutes] = f.Duration_HH_MM.split(':').map(Number);
                      return acc + hours + minutes/60;
                    }, 0) / flights.length * 10) / 10 : 0}h
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Flights Table */}
      <Card className="shadow-elegant">
        <CardHeader>
          <CardTitle>Flight Schedule</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Flight</TableHead>
                <TableHead>Route</TableHead>
                <TableHead>Departure</TableHead>
                <TableHead>Arrival</TableHead>
                <TableHead>Duration</TableHead>
                <TableHead>Aircraft</TableHead>
                <TableHead>Date</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredFlights.map((flight, index) => (
                <TableRow key={index} className="hover:bg-muted/50">
                  <TableCell className="font-medium">
                    <div className="flex items-center gap-2">
                      <Plane className="h-4 w-4 text-primary" />
                      {flight.Flight_Number}
                    </div>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline">{flight.Origin}</Badge>
                      <ArrowRight className="h-3 w-3 text-muted-foreground" />
                      <Badge variant="outline">{flight.Destination}</Badge>
                    </div>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <Clock className="h-3 w-3 text-muted-foreground" />
                      {formatTime(flight.Scheduled_Departure_UTC)}
                    </div>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <Clock className="h-3 w-3 text-muted-foreground" />
                      {formatTime(flight.Scheduled_Arrival_UTC)}
                    </div>
                  </TableCell>
                  <TableCell>
                    <Badge variant="secondary">{flight.Duration_HH_MM}</Badge>
                  </TableCell>
                  <TableCell>
                    <Badge className={getAircraftColor(flight.Aircraft_Type)}>
                      {flight.Aircraft_Type}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <Calendar className="h-3 w-3 text-muted-foreground" />
                      {flight.Date}
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between mt-4">
              <p className="text-sm text-muted-foreground">
                Showing {((currentPage - 1) * flightsPerPage) + 1} to {Math.min(currentPage * flightsPerPage, totalFlights)} of {totalFlights} flights
              </p>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                  disabled={currentPage === 1}
                >
                  Previous
                </Button>
                <span className="text-sm">
                  Page {currentPage} of {totalPages}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                  disabled={currentPage === totalPages}
                >
                  Next
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {filteredFlights.length === 0 && !loading && (
        <Card>
          <CardContent className="text-center py-8">
            <Plane className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-muted-foreground">No flights found matching your criteria</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default FlightsList;