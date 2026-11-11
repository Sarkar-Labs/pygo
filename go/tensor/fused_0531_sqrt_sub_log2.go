package tensor

import "fmt"

func Fused0531SqrtSubLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_sub_log2: nil operand")
	}
	r := x.Sub(y)
	r = r.Sqrt()
	r = r.Log2()
	return r, nil
}
func Fused0531SqrtSubLog2Name() string { return "sqrt_sub_log2" }
