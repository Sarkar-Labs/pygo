package tensor

import "fmt"

func Fused0533SqrtDivLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_div_log2: nil operand")
	}
	r := x.Div(y)
	r = r.Sqrt()
	r = r.Log2()
	return r, nil
}
func Fused0533SqrtDivLog2Name() string { return "sqrt_div_log2" }
