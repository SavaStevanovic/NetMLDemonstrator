import { Component, OnInit, Input } from '@angular/core';
import { Filter } from 'src/app/models/filter';
import { FilterService } from '../../services/filter/filter.service';

@Component({
  selector: 'app-filter',
  templateUrl: './filter.component.html',
  styleUrls: ['./filter.component.css']
})
export class FilterComponent implements OnInit {
  @Input() filter: Filter;

  constructor(private filterService: FilterService) { }

  ngOnInit(): void {
  }

  showModels(filter: Filter): void {
    filter.selected = !filter.selected;
  }

  selectModels(model: string): void {
    if (model == this.filter.selectedModel)
    {
      this.filter.selectedModel=null;
    }
    else
    {
      this.filter.selectedModel = model;
      this.filter.selected = false;
    }
  }
}
