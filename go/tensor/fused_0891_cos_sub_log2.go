package tensor

import "fmt"

func Fused0891CosSubLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_sub_log2: nil operand")
	}
	r := x.Sub(y)
	r = r.Cos()
	r = r.Log2()
	return r, nil
}
func Fused0891CosSubLog2Name() string { return "cos_sub_log2" }
