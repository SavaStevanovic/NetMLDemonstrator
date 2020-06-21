import { Component, OnInit } from '@angular/core';
import { Filter } from '../filter';

@Component({
  selector: 'app-filters',
  templateUrl: './filters.component.html',
  styleUrls: ['./filters.component.css']
})
export class FiltersComponent implements OnInit {
  filter: Filter = {
    id: 0,
    name: 'Detection',
    description: 'Bounding box detector',
    selected: true
  };
  constructor() { }

  ngOnInit(): void {
  }

}
