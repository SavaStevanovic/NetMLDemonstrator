import { Injectable } from '@angular/core';
import { Filter } from '../../models/filter';
import { Observable, Subject } from 'rxjs';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject } from 'rxjs';


@Injectable({
  providedIn: 'root'
})
export class FilterService {
  private filtersSubject = new BehaviorSubject<Filter[]>([]);
  domainSubject = new BehaviorSubject<string>("");

  constructor(private http: HttpClient) {}

  fetchFilters(url) {
    this.http.get<Filter[]>(url).subscribe(
      (filters)=>{
        this.filtersSubject.next(filters);
      }
    )
  }

  setDomain(value: string): void {
    this.domainSubject.next(value)
  }

  getDomain(): Observable<string> {
    return this.domainSubject.asObservable();
  }

  getFilters(): Observable<Filter[]> {
    return this.filtersSubject.asObservable();
  }
}
