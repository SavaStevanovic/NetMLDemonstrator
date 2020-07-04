import { Component, OnInit } from '@angular/core';
import { Observable, of } from 'rxjs';
import { Filter } from '../filter';
import { FILTERS } from '../mock-filters';
import { FilterService } from '../filter.service';

@Component({
  selector: 'app-filters',
  templateUrl: './filters.component.html',
  styleUrls: ['./filters.component.css']
})
export class FiltersComponent implements OnInit {

  filters: Filter[];

  constructor(private filterService:FilterService) { }

  ngOnInit(): void {
    this.getHeroes();
  }

  getHeroes(): void {
    this.filterService.getFilters()
      .subscribe(filters => this.filters = filters);
  }

  onSelect(filter: Filter): void {
    filter.selected = !filter.selected;
  }
}
