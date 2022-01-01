import { Injectable } from '@angular/core';
import { Filter } from '../../models/filter';
import { Observable, of } from 'rxjs';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { BehaviorSubject } from 'rxjs';
import { environment } from '../../../environments/environment';


@Injectable({
  providedIn: 'root'
})
export class FilterService {
  private filtersSubject = new BehaviorSubject<Filter[]>([]);

  constructor(private http: HttpClient) {}

  fetchFilters(url) {
    this.http.get<Filter[]>(url).subscribe(
      (filters)=>{
        this.filtersSubject.next(filters);
      }
    )
  }


  getFilters(): Observable<Filter[]> {
    return this.filtersSubject.asObservable();
  }
}
