import { Injectable } from '@angular/core';
import { Filter } from '../../models/filter';
import { Observable, of } from 'rxjs';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Subject } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class FilterService {

  private filtersUrl = 'http://127.0.0.1:4321/get_filters';
  // private filters: Filter[];
  private filtersSubject = new Subject<Filter[]>();

  constructor(private http: HttpClient) {
    this.http.get<Filter[]>(this.filtersUrl).subscribe(
      (filters)=>{
        // this.filters = filters;
        this.filtersSubject.next(filters);
      }
    );
  }

  getFilters(): Observable<Filter[]> {
    return this.filtersSubject.asObservable();
  }

  // setActiveFilter(filterName, activeFilter): void {
  //   // filters =
  //   let updateItem = this.filters.find(this.findIndexToUpdate, filterName);

  //   let index = this.filters.indexOf(updateItem);
  //   this.filters[index].selectedModel = activeFilter;
  // }

  // findIndexToUpdate(item) {
  //   return item.name === this;
  // }
}
