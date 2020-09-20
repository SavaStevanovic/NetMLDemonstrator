
export interface ProgressBar  {
  name: string,
  value: number,
}

export interface CheckBox  {
  name: string,
  checked: boolean,
}

export class Filter {

  constructor(
    public id: number,
    public name: string,
    public description: string,
    public selected: boolean,
    public models: string[],
    public selectedModel: string,
    public progress_bars: ProgressBar[],
    public check_boxes: CheckBox[],
  ) {
  }

  getName(): string {
    return this.name;
  }

  // toJSON is automatically used by JSON.stringify
  toJSON(): FilterJson {
    // copy all fields from `this` to an empty object and return in
    return Object.assign({}, this, {});
  }

  // fromJSON is used to convert an serialized version
  // of the User to an instance of the class
  static fromJSON(json: FilterJson|string): Filter {
    if (typeof json === 'string') {
      // if it's a string, parse it first
      return JSON.parse(json, Filter.reviver);
    } else {
      // create an instance of the User class
      let user = Object.create(Filter.prototype);
      // copy all the fields from the json object
      return Object.assign(user, json, {
        // convert fields that need converting
      });
    }
  }

  // reviver can be passed as the second parameter to JSON.parse
  // to automatically call User.fromJSON on the resulting value.
  static reviver(key: string, value: any): any {
    return key === "" ? Filter.fromJSON(value) : value;
  }
}

// A representation of User's data that can be converted to
// and from JSON without being altered.
export interface FilterJson {
  id: number;
  name: string;
  description: string;
  selected: boolean;
  models: string[];
  selectedModel: string;
}
