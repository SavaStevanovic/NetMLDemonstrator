import { Component, OnInit } from '@angular/core';
import { Filter } from '../../models/filter';
import { FilterService } from '../../services/filter/filter.service';

@Component({
  selector: 'app-filters',
  templateUrl: './filters.component.html',
  styleUrls: ['./filters.component.css']
})

export class FiltersComponent implements OnInit {

  filters: Filter[];

  constructor(private filterService:FilterService) { }

  ngOnInit(): void {
    this.getFilters();
  }

  getFilters(): void {
    this.filterService.getFilters()
      .subscribe(filters => this.filters = filters);
  }
}
