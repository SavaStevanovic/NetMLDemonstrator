import { Component, OnInit } from '@angular/core';
import { Filter } from '../../models/filter';
import { FilterService } from '../../services/filter/filter.service';
import { environment } from '../../../environments/environment';
import { HttpClient, HttpHeaders } from '@angular/common/http';

@Component({
  selector: 'app-filters',
  templateUrl: './filters.component.html',
  styleUrls: ['./filters.component.css']
})

export class FiltersComponent implements OnInit {

  filters: Filter[];
  selectedDomain: string;

  constructor(private filterService:FilterService) {
    this.domains = Object.keys(environment.domains)
    this.selectedDomain = this.domains[0]
  }

  ngOnInit(): void {
    this.changeDomain()
  }

  changeDomain(): void {
    this.filterService.fetchFilters(environment.domains[this.selectedDomain].get_filters)
    this.getFilters();
  }

  getFilters(): void {
    this.filterService.getFilters()
      .subscribe(filters => this.filters = filters.filter(f => !environment.production || !f.name.includes('Test')));
  }
}
