package tensor

import "fmt"

func Fused0933FloorDivLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_div_log: nil operand")
	}
	r := x.Div(y)
	r = r.Floor()
	r = r.Log()
	return r, nil
}
func Fused0933FloorDivLogName() string { return "floor_div_log" }
