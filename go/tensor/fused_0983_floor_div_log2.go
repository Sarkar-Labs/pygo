package tensor

import "fmt"

func Fused0983FloorDivLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_div_log2: nil operand")
	}
	r := x.Div(y)
	r = r.Floor()
	r = r.Log2()
	return r, nil
}
func Fused0983FloorDivLog2Name() string { return "floor_div_log2" }
