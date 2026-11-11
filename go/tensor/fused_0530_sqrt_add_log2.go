package tensor

import "fmt"

func Fused0530SqrtAddLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_add_log2: nil operand")
	}
	r := x.Add(y)
	r = r.Sqrt()
	r = r.Log2()
	return r, nil
}
func Fused0530SqrtAddLog2Name() string { return "sqrt_add_log2" }
