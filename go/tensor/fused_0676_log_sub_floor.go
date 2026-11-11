package tensor

import "fmt"

func Fused0676LogSubFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_sub_floor: nil operand")
	}
	r := x.Sub(y)
	r = r.Log()
	r = r.Floor()
	return r, nil
}
func Fused0676LogSubFloorName() string { return "log_sub_floor" }
