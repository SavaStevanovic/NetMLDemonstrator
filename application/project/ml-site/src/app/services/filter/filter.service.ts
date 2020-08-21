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
  private filtersSubject = new Subject<Filter[]>();

  constructor(private http: HttpClient) {
    this.http.get<Filter[]>(this.filtersUrl).subscribe(
      (filters)=>{
        this.filtersSubject.next(filters);
      }
    );
  }

  getFilters(): Observable<Filter[]> {
    return this.filtersSubject.asObservable();
  }
}
