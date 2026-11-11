package tensor

import "fmt"

func Fused1611Log2SubLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_sub_log2: nil operand")
	}
	r := x.Sub(y)
	r = r.Log2()
	r = r.Log2()
	return r, nil
}
func Fused1611Log2SubLog2Name() string { return "log2_sub_log2" }
