export interface Filter {
  id: number;
  name: string;
  description: string;
  selected: boolean;
  models: string[];
  selectedModel: string;
  thresholdValue: number;
}
