package tensor

import "fmt"

func Fused0988FloorDivLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_div_log10: nil operand")
	}
	r := x.Div(y)
	r = r.Floor()
	r = r.Log10()
	return r, nil
}
func Fused0988FloorDivLog10Name() string { return "floor_div_log10" }
