import { Injectable } from '@angular/core';
import { Filter } from '../../models/filter';
import { Observable, of } from 'rxjs';
import { HttpClient, HttpHeaders } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class FilterService {

  private filtersUrl = 'http://127.0.0.1:4321/get_filters';

  constructor(private http: HttpClient) { }

  getFilters(): Observable<Filter[]> {
    return this.http.get<Filter[]>(this.filtersUrl);
  }
}
