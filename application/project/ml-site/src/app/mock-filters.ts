import { Filter } from './filter'

export const FILTERS: Filter[] = [
  { id: 0, name: 'Detection', description: 'Bounding box detector', selected: true },
  { id: 1, name: 'Segmentation', description: 'Placeholer Segmentation', selected: false },
  { id: 2, name: 'Superresolution', description: 'Placeholer Superresolution', selected: false },
  { id: 3, name: 'Recognition', description: 'Placeholer Recognition', selected: false },
  { id: 4, name: 'Image To Text', description: 'Placeholer Image To Text', selected: false },
];
