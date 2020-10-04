import { Injectable } from '@angular/core';
import { Filter } from '../../models/filter';
import { Observable, of } from 'rxjs';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Subject } from 'rxjs';
import { environment } from '../../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class FilterService {

  private filtersUrl = environment.filtersUrl;
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
