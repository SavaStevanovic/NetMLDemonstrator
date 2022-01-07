import { Component, OnInit } from '@angular/core';
import { Filter } from '../../models/filter';
import { FilterService } from '../../services/filter/filter.service';
import { environment } from '../../../environments/environment';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { StateService } from 'src/app/services/state/state.service';

@Component({
  selector: 'app-filters',
  templateUrl: './filters.component.html',
  styleUrls: ['./filters.component.css']
})

export class FiltersComponent implements OnInit {

  domains: string[] = Object.keys(environment.domains);
  filters: Filter[];
  selectedDomain: string = this.domains[0];

  constructor(
    private filterService:FilterService,
    private stateService: StateService) {}

  ngOnInit(): void {
    this.changeDomain()
  }

  changeDomain(): void {
    this.stateService.setVideoStart(false)
    this.filterService.fetchFilters(environment.domains[this.selectedDomain].get_filters)
    this.filterService.setDomain(this.selectedDomain)
    this.getFilters();
  }

  getFilters(): void {
    this.filterService.getFilters()
      .subscribe(filters => this.filters = filters.filter(f => !environment.production || !f.name.includes('Test')));
  }
}
