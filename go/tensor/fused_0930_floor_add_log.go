package tensor

import "fmt"

func Fused0930FloorAddLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_add_log: nil operand")
	}
	r := x.Add(y)
	r = r.Floor()
	r = r.Log()
	return r, nil
}
func Fused0930FloorAddLogName() string { return "floor_add_log" }
