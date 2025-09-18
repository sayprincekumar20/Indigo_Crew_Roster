import React, { useState, useRef, useEffect } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { 
  MessageSquare, 
  Send, 
  Bot, 
  User, 
  AlertTriangle,
  CheckCircle,
  Loader2,
  RefreshCw,
  Plus,
  X,
  Calendar,
  User as UserIcon,
  Plane,
  CloudRain,
  Wrench,
  Shield,

} from "lucide-react";
import { toast } from "sonner";

interface ChatMessage {
  id: string;
  type: 'user' | 'bot' | 'system';
  content: string;
  timestamp: Date;
  suggestedActions?: string[];
  data?: any;
}

interface DisruptionAnalysis {
  status: string;
  analysis: string;
  affected_flights: number;
  detailed_analysis: any[];
}

interface DisruptionChatbotProps {
  isOpen: boolean;
  onClose: () => void;
}

const DisruptionChatbot: React.FC<DisruptionChatbotProps> = ({ isOpen, onClose }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      type: 'bot',
      content: 'Hello! I\'m your IndiGo Disruption Management Assistant. I can help you handle crew disruptions, find replacements, and ensure DGCA compliance. How can I assist you today?',
      timestamp: new Date(),
      suggestedActions: ['report_disruption', 'view_roster', 'check_compliance']
    }
  ]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showDisruptionForm, setShowDisruptionForm] = useState(false);
  const [disruptionForm, setDisruptionForm] = useState({
    crew_id: '',
    flight_number: '',
    start_date: '',
    end_date: '',
    reason: '',
    disruption_type: 'sickness'
  });
  const [currentDisruption, setCurrentDisruption] = useState<any>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Example scenarios for quick testing
  const exampleScenarios = [
    {
      title: "Crew Sickness",
      description: "JCC001 called in sick",
      message: "JCC001 called in sick for flight 6E101 on 2023-10-01",
      icon: <UserIcon className="h-4 w-4" />,
      formData: {
        crew_id: 'JCC001',
        flight_number: '6E101',
        start_date: '2023-10-01',
        end_date: '2023-10-01',
        reason: 'illness',
        disruption_type: 'sickness'
      }
    },
    {
      title: "Weather Disruption",
      description: "Bad weather in Delhi",
      message: "Bad weather in DEL is affecting multiple flights today",
      icon: <CloudRain className="h-4 w-4" />,
      formData: {
        crew_id: '',
        flight_number: '',
        start_date: new Date().toISOString().split('T')[0],
        end_date: new Date().toISOString().split('T')[0],
        reason: 'weather conditions',
        disruption_type: 'weather'
      }
    }
  ];

  // Scroll to bottom of chat
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleSendMessage = async () => {
    if (!currentMessage.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: currentMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setCurrentMessage('');
    setIsLoading(true);

    try {
      const response = await fetch('http://127.0.0.1:8000/api/v1/disruption/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: currentMessage,
          context: {}
        }),
      });

      const data = await response.json();

      const botMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: data.response,
        timestamp: new Date(),
        suggestedActions: data.suggested_actions
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      toast.error('Failed to send message');
      
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: 'I apologize, but I\'m having trouble processing your request right now. Please try again later.',
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleAnalyzeDisruption = async () => {
    if (!disruptionForm.crew_id || !disruptionForm.start_date) {
      toast.error('Please provide at least Crew ID and Start Date');
      return;
    }

    setIsLoading(true);
    
    // Add user message showing what they're reporting
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: `Report disruption for crew ${disruptionForm.crew_id} from ${disruptionForm.start_date} to ${disruptionForm.end_date || disruptionForm.start_date}: ${disruptionForm.reason}`,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setShowDisruptionForm(false);

    try {
      const response = await fetch('http://127.0.0.1:8000/api/v1/disruption/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          crew_id: disruptionForm.crew_id,
          flight_number: disruptionForm.flight_number || null,
          start_date: disruptionForm.start_date,
          end_date: disruptionForm.end_date || disruptionForm.start_date,
          reason: disruptionForm.reason,
          disruption_type: disruptionForm.disruption_type
        }),
      });

      const data: DisruptionAnalysis = await response.json();

      if (data.status === 'success') {
        const botMessage: ChatMessage = {
          id: (Date.now() + 1).toString(),
          type: 'bot',
          content: data.analysis,
          timestamp: new Date(),
          data: {
            ...data,
            originalCrewId: disruptionForm.crew_id
          }
          
        };

        setMessages(prev => [...prev, botMessage]);

      setCurrentDisruption({
         ...data,
            originalCrewId: disruptionForm.crew_id
      });
        
        // Reset form
        setDisruptionForm({
          crew_id: '',
          flight_number: '',
          start_date: '',
          end_date: '',
          reason: '',
          disruption_type: 'sickness'
        });
      } else {
        throw new Error(data.error || 'Failed to analyze disruption');
      }
    } catch (error) {
      console.error('Error analyzing disruption:', error);
      toast.error('Failed to analyze disruption');
      
      const errorMessage: ChatMessage = {
        id: (Date.now() + 2).toString(),
        type: 'bot',
        content: 'I apologize, but I\'m having trouble analyzing this disruption. Please try again later.',
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

const handleApplyReplacement = async (flightNumber: string, originalCrew: string, newCrew: string) => {
  // Add validation to ensure originalCrew is not empty
  if (!originalCrew || originalCrew.trim() === '') {
    toast.error('Original crew ID is required');
    return;
  }
  
  setIsLoading(true);
  
  try {
    console.log('Sending replacement payload:', {
      original_crew: originalCrew,
      replacements: [
        {
          flight_number: flightNumber,
          new_crew: newCrew
        }
      ]
    });

    const response = await fetch('http://127.0.0.1:8000/api/v1/disruption/apply-replacement', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        original_crew: originalCrew,
        replacements: [
          {
            flight_number: flightNumber,
            new_crew: newCrew
          }
        ]
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    const data = await response.json();

    if (data.status === 'success') {
      toast.success('Replacement applied successfully');
      
      const systemMessage: ChatMessage = {
        id: Date.now().toString(),
        type: 'system',
        content: `Replacement applied: ${originalCrew} → ${newCrew} for flight ${flightNumber}`,
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, systemMessage]);
      
      // Refresh the disruption analysis
      if (currentDisruption) {
        const analyzeMessage: ChatMessage = {
          id: Date.now().toString(),
          type: 'user',
          content: `Re-analyze disruption after replacement of ${originalCrew} with ${newCrew}`,
          timestamp: new Date()
        };
        
        setMessages(prev => [...prev, analyzeMessage]);
        handleAnalyzeDisruption();
      }
    } else {
      throw new Error(data.message || 'Failed to apply replacement');
    }
  } catch (error) {
    console.error('Error applying replacement:', error);
    toast.error('Failed to apply replacement: ' + (error instanceof Error ? error.message : 'Unknown error'));
  } finally {
    setIsLoading(false);
  }
};
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleSuggestedAction = (action: string) => {
    const actionMessages = {
      'report_disruption': 'I want to report a crew disruption',
      'analyze_disruption': 'Please analyze the current disruption situation and provide recommendations.',
      'view_roster': 'Show me the current roster status and any conflicts.',
      'check_compliance': 'Check DGCA compliance for the current roster assignments.',
      'find_replacement': 'Find a replacement crew member',
      'adjust_schedule': 'Adjust the flight schedule',
      'notify_crew': 'Notify affected crew members',
      'notify_passengers': 'Notify passengers about changes'
    };

    if (action === 'report_disruption') {
      setShowDisruptionForm(true);
      return;
    }

    const message = actionMessages[action as keyof typeof actionMessages] || action;
    setCurrentMessage(message);
  };

  const handleExampleScenario = (scenario: any) => {
    if (scenario.formData) {
      setDisruptionForm(scenario.formData);
      setShowDisruptionForm(true);
    } else {
      setCurrentMessage(scenario.message);
    }
  };

  const clearChat = () => {
    setMessages([
      {
        id: '1',
        type: 'bot',
        content: 'Hello! I\'m your IndiGo Disruption Management Assistant. I can help you handle crew disruptions, find replacements, and ensure DGCA compliance. How can I assist you today?',
        timestamp: new Date(),
        suggestedActions: ['report_disruption', 'view_roster', 'check_compliance']
      }
    ]);
    setShowDisruptionForm(false);
    setCurrentDisruption(null);
  };

  const formatBotMessage = (content: string) => {
    if (!content) return '';
    return content
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/\n/g, '<br />');
  };

  const renderDisruptionForm = () => (
    <Card className="mb-4">
      <CardContent className="p-4">
        <div className="flex justify-between items-center mb-4">
          <h3 className="font-semibold">Report Crew Disruption</h3>
          <Button 
            variant="ghost" 
            size="sm" 
            onClick={() => setShowDisruptionForm(false)}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="space-y-2">
            <label className="text-sm font-medium">Crew ID *</label>
            <Input
              placeholder="e.g., JCC001"
              value={disruptionForm.crew_id}
              onChange={(e) => setDisruptionForm({...disruptionForm, crew_id: e.target.value})}
            />
          </div>
          
          <div className="space-y-2">
            <label className="text-sm font-medium">Flight Number</label>
            <Input
              placeholder="e.g., 6E101"
              value={disruptionForm.flight_number}
              onChange={(e) => setDisruptionForm({...disruptionForm, flight_number: e.target.value})}
            />
          </div>
          
          <div className="space-y-2">
            <label className="text-sm font-medium">Start Date *</label>
            <Input
              type="date"
              value={disruptionForm.start_date}
              onChange={(e) => setDisruptionForm({...disruptionForm, start_date: e.target.value})}
            />
          </div>
          
          <div className="space-y-2">
            <label className="text-sm font-medium">End Date</label>
            <Input
              type="date"
              value={disruptionForm.end_date}
              onChange={(e) => setDisruptionForm({...disruptionForm, end_date: e.target.value})}
            />
          </div>
          
          <div className="space-y-2 md:col-span-2">
            <label className="text-sm font-medium">Reason *</label>
            <select
              className="w-full p-2 border rounded-md"
              value={disruptionForm.disruption_type}
              onChange={(e) => setDisruptionForm({...disruptionForm, disruption_type: e.target.value})}
            >
              <option value="sickness">Sickness/Medical</option>
              <option value="technical">Technical Issue</option>
              <option value="weather">Weather</option>
              <option value="personal">Personal Reasons</option>
              <option value="other">Other</option>
            </select>
          </div>
          
          <div className="space-y-2 md:col-span-2">
            <label className="text-sm font-medium">Additional Details</label>
            <Textarea
              placeholder="Provide additional details about the disruption..."
              value={disruptionForm.reason}
              onChange={(e) => setDisruptionForm({...disruptionForm, reason: e.target.value})}
              rows={2}
            />
          </div>
        </div>
        
        <Button 
          onClick={handleAnalyzeDisruption} 
          className="mt-4 w-full"
          disabled={!disruptionForm.crew_id || !disruptionForm.start_date || isLoading}
        >
          {isLoading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin mr-2" />
              Analyzing...
            </>
          ) : (
            'Analyze Disruption'
          )}
        </Button>
      </CardContent>
    </Card>
  );

  const renderAnalysisResult = (message: ChatMessage) => {
    if (!message.data) return null;
    
    const analysis = message.data as DisruptionAnalysis;
    
    return (
      <div className="mt-4 space-y-4">
        <div className="p-3 bg-blue-50 rounded-lg">
          <h4 className="font-semibold text-blue-800">Analysis Summary</h4>
          <p className="text-sm text-blue-700 mt-1">{analysis.affected_flights} affected flight(s) found</p>
        </div>
        
        {analysis.detailed_analysis && analysis.detailed_analysis.map((flightAnalysis, index) => (
          <Card key={index} className="bg-muted/30">
            <CardContent className="p-4">
              <div className="flex items-center justify-between mb-3">
                <div>
                  <h4 className="font-semibold">Flight {flightAnalysis.flight?.Flight_Number}</h4>
                  {flightAnalysis.flight && (
                    <p className="text-sm text-muted-foreground">
                      {flightAnalysis.flight.Origin} → {flightAnalysis.flight.Destination} • {flightAnalysis.flight.Date}
                    </p>
                  )}
                </div>
                {flightAnalysis.flight && (
                  <Badge variant="outline" className="flex items-center gap-1">
                    <Plane className="h-3 w-3" />
                    {flightAnalysis.flight.Aircraft_Type}
                  </Badge>
                )}
              </div>
              
              {flightAnalysis.llm_analysis && (
                <div className="mb-4 p-3 bg-muted rounded-md">
                  <h5 className="text-sm font-medium mb-2">Detailed Analysis</h5>
                  <div 
                    className="text-sm"
                    dangerouslySetInnerHTML={{ __html: formatBotMessage(flightAnalysis.llm_analysis) }}
                  />
                </div>
              )}
              
              {flightAnalysis.candidates && (
                <div>
                  <h5 className="text-sm font-medium mb-2">Recommended Replacements</h5>
                  <div className="space-y-3">
                    {flightAnalysis.candidates.map((candidate: any, i: number) => (
                      <Card key={i} className="p-3">
                        <div className="flex justify-between items-start">
                          <div>
                            <p className="font-medium">{candidate.crew_id} • {candidate.name}</p>
                            <p className="text-xs text-muted-foreground">{candidate.rank} • {candidate.base}</p>
                            {candidate.qualifications && (
                              <p className="text-xs text-muted-foreground">Qualifications: {candidate.qualifications}</p>
                            )}
                            <div className="flex items-center mt-1">
                              <Badge variant={candidate.score > 80 ? "default" : "secondary"}>
                                Score: {candidate.score}%
                              </Badge>
                            </div>
                          </div>
                          {flightAnalysis.flight && (
                            <Button 
                              size="sm"
                              onClick={() => handleApplyReplacement(
                                flightAnalysis.flight.Flight_Number, 
                                currentDisruption.originalCrewId, 
                                candidate.crew_id
                              )}
                              disabled={isLoading}
                            >
                              {isLoading ? (
                                <Loader2 className="h-4 w-4 animate-spin" />
                              ) : (
                                'Apply'
                              )}
                            </Button>
                          )}
                        </div>
                        
                        <div className="mt-2">
                          {candidate.reasons && candidate.reasons.length > 0 && (
                            <div className="text-xs">
                              <span className="font-medium">Reasons: </span>
                              {candidate.reasons.join(', ')}
                            </div>
                          )}
                          
                          {candidate.warnings && candidate.warnings.length > 0 && (
                            <div className="text-xs text-amber-600 mt-1">
                              <span className="font-medium">Warnings: </span>
                              {candidate.warnings.join(', ')}
                            </div>
                          )}
                          
                          {candidate.compliance_check && (
                            <div className="mt-2">
                              <h6 className="text-xs font-medium">DGCA Compliance:</h6>
                              <div className={`text-xs ${candidate.compliance_check.all_ok ? 'text-green-600' : 'text-red-600'}`}>
                                {candidate.compliance_check.all_ok ? 
                                  '✓ Fully compliant' : 
                                  '⚠️ Potential compliance issues'
                                }
                              </div>
                              {candidate.compliance_check.issues && candidate.compliance_check.issues.length > 0 && (
                                <ul className="text-xs text-red-600 mt-1">
                                  {candidate.compliance_check.issues.map((issue: string, idx: number) => (
                                    <li key={idx}>• {issue}</li>
                                  ))}
                                </ul>
                              )}
                            </div>
                          )}
                        </div>
                      </Card>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        ))}
      </div>
    );
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl h-[90vh] flex flex-col">
        <DialogHeader className="flex flex-row items-center justify-between">
          <DialogTitle className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-primary/10">
              <MessageSquare className="h-6 w-6 text-primary" />
            </div>
            Disruption Management Assistant
          </DialogTitle>
          <Button 
            variant="outline" 
            size="sm" 
            onClick={clearChat}
            className="gap-2"
          >
            <RefreshCw className="h-4 w-4" />
            Clear Chat
          </Button>
        </DialogHeader>

        <div className="flex-1 overflow-hidden flex flex-col">
          {/* Chat Messages */}
          <div className="flex-1 overflow-y-auto space-y-4 p-4 border rounded-lg bg-muted/20 mb-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex gap-3 ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                {message.type === 'bot' && (
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
                    <Bot className="h-4 w-4 text-primary" />
                  </div>
                )}
                
                {message.type === 'system' && (
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-green-100 flex items-center justify-center">
                    <CheckCircle className="h-4 w-4 text-green-600" />
                  </div>
                )}
                
                <div className={`max-w-[80%] ${message.type === 'user' ? 'order-2' : ''}`}>
                  <Card className={
                    message.type === 'user' ? 'bg-primary text-primary-foreground' : 
                    message.type === 'system' ? 'bg-green-50 border-green-200' : ''
                  }>
                    <CardContent className="p-3">
                      <div 
                        className="text-sm"
                        dangerouslySetInnerHTML={{ 
                          __html: (message.type === 'bot' || message.type === 'system') ? 
                            formatBotMessage(message.content) : message.content 
                        }}
                      />
                      <div className="text-xs opacity-70 mt-2">
                        {message.timestamp.toLocaleTimeString()}
                      </div>
                    </CardContent>
                  </Card>
                  
                  {/* Render analysis results if available */}
                  {message.type === 'bot' && message.data && renderAnalysisResult(message)}
                  
                  {/* Suggested Actions */}
                  {message.suggestedActions && message.suggestedActions.length > 0 && (
                    <div className="mt-2 space-y-1">
                      <p className="text-xs text-muted-foreground">Suggested actions:</p>
                      <div className="flex flex-wrap gap-1">
                        {message.suggestedActions.map((action, index) => (
                          <Button
                            key={index}
                            variant="outline"
                            size="sm"
                            onClick={() => handleSuggestedAction(action)}
                            className="text-xs h-7"
                          >
                            {action.replace(/_/g, ' ')}
                          </Button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {message.type === 'user' && (
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary flex items-center justify-center order-1">
                    <User className="h-4 w-4 text-primary-foreground" />
                  </div>
                )}
              </div>
            ))}
            
            {isLoading && (
              <div className="flex gap-3 justify-start">
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
                  <Bot className="h-4 w-4 text-primary" />
                </div>
                <Card>
                  <CardContent className="p-3">
                    <div className="flex items-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      <span className="text-sm">Analyzing your request...</span>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          {/* Disruption Form or Message Input */}
          {showDisruptionForm ? (
            renderDisruptionForm()
          ) : (
            <>
              {/* Message Input */}
              <div className="flex gap-2">
                <Input
                  value={currentMessage}
                  onChange={(e) => setCurrentMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Type your disruption query here... (e.g., 'JCC001 called in sick for flight 6E101')"
                  disabled={isLoading}
                  className="flex-1"
                />
                <Button 
                  onClick={handleSendMessage} 
                  disabled={!currentMessage.trim() || isLoading}
                  className="gap-2"
                >
                  {isLoading ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <>
                      <Send className="h-4 w-4" />
                      Send
                    </>
                  )}
                </Button>
              </div>

              {/* Quick Actions */}
              <div className="mt-3 p-3 border rounded-lg bg-muted/20">
                <p className="text-xs font-medium text-muted-foreground mb-2">Quick Actions:</p>
                <div className="flex flex-wrap gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setShowDisruptionForm(true)}
                    className="text-xs h-7 gap-1"
                  >
                    <Plus className="h-3 w-3" />
                    Report Disruption
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleExampleScenario(exampleScenarios[0])}
                    className="text-xs h-7"
                  >
                    Report Sickness
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setCurrentMessage('Find replacement for JCC001 on flight 6E101')}
                    className="text-xs h-7"
                  >
                    Find Replacement
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setCurrentMessage('Check DGCA compliance for current roster')}
                    className="text-xs h-7"
                  >
                    Check Compliance
                  </Button>
                </div>
                
                {/* Example Scenarios */}
                <p className="text-xs font-medium text-muted-foreground mt-4 mb-2">Try These Examples:</p>
                <div className="grid grid-cols-2 gap-2">
                  {exampleScenarios.map((scenario, index) => (
                    <Card 
                      key={index} 
                      className="p-2 cursor-pointer hover:bg-accent transition-colors"
                      onClick={() => handleExampleScenario(scenario)}
                    >
                      <CardContent className="p-0 flex items-center gap-2">
                        <div className="text-muted-foreground">
                          {scenario.icon}
                        </div>
                        <div className="flex-1">
                          <p className="text-xs font-semibold">{scenario.title}</p>
                          <p className="text-xs text-muted-foreground">{scenario.description}</p>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default DisruptionChatbot;