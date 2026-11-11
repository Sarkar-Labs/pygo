package tensor

import "fmt"

func Fused0443AbsDivLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_div_log2: nil operand")
	}
	r := x.Div(y)
	r = r.Abs()
	r = r.Log2()
	return r, nil
}
func Fused0443AbsDivLog2Name() string { return "abs_div_log2" }
